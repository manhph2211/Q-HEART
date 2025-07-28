import glob
import os
import scipy
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
import json
import random
from models.ecg_encoder.ecg_encoder import get_ecg_feats, get_model, get_batch_ecg_feats
import scipy.io as sio
import pandas as pd 


ecg_encoder = get_model()
for param in ecg_encoder.parameters():
    param.requires_grad = False
ecg_encoder.eval()


def process_text(text):
    text = text.lower().strip()
    report = text.replace('ekg', 'ecg')
    report = text.replace('ekg', 'ecg').replace("1st","first")
    report = report.strip('*** ').strip(' ***').strip('***').strip('=-').strip('=')
    return report


def process_file(mat_file):
    try:
        data = scipy.io.loadmat(mat_file)
        text = data["text"][0]
        ecg = data["feats"]
        return ecg, process_text(text)
    except:
        print(mat_file)
        return None


def get_unique_text_ecg(ecg_data, text_data):
    unique_dict = {}
    for ecg, text in tqdm(zip(ecg_data, text_data)):
        if text not in unique_dict:
            unique_dict[text] = ecg
    unique_texts = list(unique_dict.keys())
    unique_ecgs = list(unique_dict.values())
    return unique_ecgs, unique_texts


def faiss_write(ecg_encoder, data_root="data/processed_data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_encoder = ecg_encoder.to(device)

    mat_files = glob.glob(os.path.join(data_root,"*.mat"))[:150000]
    print("Number of Samples: ", len(mat_files))

    ecg_data = []
    text_data = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_file, mat_file): mat_file for mat_file in mat_files}

        for future in tqdm(as_completed(futures), total=len(mat_files)):
            result = future.result()
            if result:
                ecg_data.append(result[0])
                text_data.append(result[1])

    ecg_data, batch_texts = get_unique_text_ecg(ecg_data, text_data)
    print(len(batch_texts))

    with torch.no_grad():
        batch_ecg_embeddings = get_batch_ecg_feats(ecg_encoder, ecg_data, batch_size=128, device=device).cpu().numpy()

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    batch_ecg_embeddings = normalizer(batch_ecg_embeddings)
    index = faiss.IndexFlatL2(batch_ecg_embeddings.shape[1])

    index.add(batch_ecg_embeddings)
    embedding_to_sample_map = {i: batch_texts[i] for i in range(len(batch_texts))}
    
    with open(f'data/raw/ecg_index.json', 'w') as f:
        json.dump(embedding_to_sample_map, f)
    faiss.write_index(index, f"data/raw/ecg_index.faiss")
    print("Successfully Save Files !")


def faiss_read(ecg_encoder, query_ecg, index, embedding_to_sample_map, k=3):
    embedding_to_sample_map = {int(key): v for key, v in embedding_to_sample_map.items()}

    device = "cpu"
    ecg_encoder = ecg_encoder.to(device)

    with torch.no_grad():
        query_embedding = get_ecg_feats(ecg_encoder, torch.FloatTensor(query_ecg).reshape(1,12,5000)).cpu().numpy()
        normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
        query_embedding = normalizer(query_embedding)

    distances, indices = index.search(query_embedding, k)
    similar_samples = ", ".join([embedding_to_sample_map[idx] for idx in indices[0]])
    return similar_samples


class FaissIndexing:
    def __init__(self, data_dir, csv_data_path):
        self.data_dir = data_dir
        
        self.ecg_encoder = ecg_encoder
    
        self.faiss_index = faiss.read_index(f"data/raw/ecg_index.faiss")
        with open('data/raw/ecg_index.json', 'r') as f:
            self.faiss_embedding_to_sample_map = json.load(f)
                    
        data = pd.read_csv(csv_data_path, sep='\t')
        self.ecg_ids = data['ecg_id'].values
        self.ecg_ids = np.array([str(element).zfill(5) for element in self.ecg_ids])
        self.questions = data['question'].values
        self.answers = data['answer'].values   
        self.df = data  
        self.csv_data_path = csv_data_path

    def read_sample(self, path):
        record = sio.loadmat(path)
        ecg = record["feats"]
        return ecg

    def retrieval(self):
        contexts = []
        for idx in tqdm(range(len(self.answers))):
            ecg = self.read_sample(os.path.join(self.data_dir, f"{int(self.ecg_ids[idx])}.mat"))            
            context = faiss_read(self.ecg_encoder, ecg, self.faiss_index, self.faiss_embedding_to_sample_map, k=3)            
            contexts.append(context)
        
        self.df['context'] = contexts
        
        output_file = f"{self.csv_data_path.replace('.tsv', '')}_with_context.tsv"
        self.df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved updated data with context to {output_file}")
        

if __name__ == "__main__":
    # faiss_write(ecg_encoder, "/common/home/users/h/hm.pham.2023/workspace/ecg_foundation_model/data/processed_data") 
    
    train_data_csv_path = f"data/manifest/mimic_ecg_qa/train_qa.tsv"
    val_data_csv_path = f"data/manifest/mimic_ecg_qa/valid_qa.tsv"
    test_data_csv_path = f"data/manifest/mimic_ecg_qa/test_qa.tsv"

    data_root = "/common/home/users/h/hm.pham.2023/workspace/ecg_foundation_model/data/processed_data"
    for csv_path in [test_data_csv_path, val_data_csv_path, train_data_csv_path]:
        faiss_indexing = FaissIndexing(data_root, csv_path)
        faiss_indexing.retrieval()

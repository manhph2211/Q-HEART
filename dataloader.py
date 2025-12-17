import sys
import os
import pandas as pd
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import ast


class CustomECGQADataset(Dataset):
    def __init__(self, data_dir, csv_data_path, question_type="", sample_data=None, args=None):
        
        self.data_dir = data_dir
        data = pd.read_csv(csv_data_path, sep='\t')
        
        if question_type != "all":
            data = data[data["question_type"] == question_type] 
            
        if sample_data is not None: 
            data = data.sample(n=sample_data)

        self.ecg_ids = data['ecg_id'].values
        self.ecg_ids = np.array([str(element).zfill(5) for element in self.ecg_ids])
        self.questions = data['question'].values
        self.answers = data['answer'].values   
        self.question_types = data['question_type'].values
        self.attributes =  data['attribute'].values
        self.add_context = args.add_context  
        
        self.contexts = data['context'].values 
            
    def __len__(self):
        return len(self.ecg_ids)
    
    def read_sample(self, path):
        record = sio.loadmat(path)
        ecg = record["feats"]
        return ecg
    
    def get_opts(self, attribute, question_type, question):
        if question_type == "single-verify":
            attribute = "yes, no, not sure"
        elif question_type == "single-query":
            if question.startswith("What leads"):
                attribute = "lead I, lead II, lead III, lead aVR, lead aVL, lead aVF, lead V1, lead V2, lead V3, lead V4, lead V5, lead V6"
            elif question.startswith("What numeric features"):
                attribute = "rr interval, p duration, pr interval, qrs duration, qt interval, qt corrected"
            elif question.startswith("What range"):
                attribute = "below the normal range, within the normal range, above the normal range"
            else:
                attribute += ", none"
        elif question_type == "single-choose":
            attribute += ", none"
        return attribute

    def get_random_context(self):
        return "dummy context"

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        question_type = self.question_types[idx]
        attribute = self.get_opts(self.attributes[idx], question_type, question)
        context = self.contexts[idx].split(",")[0]

        ecg = self.read_sample(os.path.join(self.data_dir, f"{int(self.ecg_ids[idx])}.mat")) 
        ecg = torch.tensor(ecg, dtype=torch.float32)
        return question, context, answer, ecg, attribute, question_type


class CustomDataCollatorQA:
    def __init__(self, model_type, add_context=True, add_options=True, instruct_template=True, shuffle_options=False, lead_care=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.instruct_template = instruct_template
        self.question_prompt = ""
        if self.instruct_template:
            self.instruct_prompt = "Given this ECG sequences, selecting suitable options strictly from" 
            self.question_prompt = "to answer this"
        else:
            self.instruct_prompt = ""
        self.add_context = add_context
        self.add_options = add_options
        self.shuffle_options = shuffle_options
        self.lead_care = lead_care
        

    def shuffle_options_(self, options_prompt):
        options_list = options_prompt.strip().split(", ")
        random.shuffle(options_list)
        return ", ".join(options_list)
    
    def lead_care_(self, question, ecg, mask_prob=0.5):
        shape = ecg.shape
        ecg = ecg.reshape(1, 12, 5000)
        lead_names = ["lead I", "lead II", "lead III", "lead aVR", "lead aVL", "lead aVF", "lead V1", "lead V2", "lead V3", "lead V4", "lead V5", "lead V6"]
        lead_to_index = {name: idx for idx, name in enumerate(lead_names)}
        masked_ecg = ecg.clone()        
        target_lead = None
        for lead in lead_names:
            if lead.lower() in question:
                target_lead = lead
                break

        if target_lead is not None:
            target_idx = lead_to_index[target_lead]
            for i in range(len(lead_names)):
                if i != target_idx:  
                    if random.random() < mask_prob:  
                        masked_ecg[:, i, :] = 0

        return masked_ecg.reshape(shape)

    def __call__(self, batch):
        questions, contexts, answers, ecg_signals, attributes, types = zip(*batch)
        custom_ecg_signals = [] 
        combined_texts = []
        for q, o, a, c, e, t in zip(questions, attributes, answers, contexts, ecg_signals, types):
            options_prompt = ""
            contexts_prompt = ""
            if self.add_options:
                shuffle_options_ = self.shuffle_options_(o.strip().lower()) if self.shuffle_options else o.strip().lower()
                options_prompt = f"({shuffle_options_})" 
            if self.add_context:
                contexts_prompt = f" Reports: {self.shuffle_options_(c.strip().lower())}."
            custom_e = self.lead_care_(q.lower(), e) if self.lead_care else e
            custom_ecg_signals.append(custom_e)
            message = f'''{self.instruct_prompt} {options_prompt} {self.question_prompt} Question: {q.lower()}{contexts_prompt} \n Answer: {a.lower()}'''
                   
            combined_texts.append(message)

        custom_ecg_signals = torch.stack([s for s in custom_ecg_signals])
        tokenized = self.tokenizer(
            combined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        labels = input_ids.clone()
        for i, text in enumerate(combined_texts):
            answer_start = text.find("Answer:")
            prefix = text[:answer_start + len("Answer:")]
            prefix_len = len(self.tokenizer(prefix).input_ids) - 1
            labels[i, :prefix_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'ecg': custom_ecg_signals,
            'prefix_len': prefix_len 
        }

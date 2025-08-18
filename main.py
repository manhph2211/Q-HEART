import sys
import os
import argparse
import numpy as np
import random
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from predict import custom_eval_llm_open_ended
from models.models import CustomECGQAModel
from dataloader import CustomECGQADataset, CustomDataCollatorQA
from train import custom_pytorch_model_run
from models.ecg_encoder.ecg_encoder import get_model


def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="meta-llama/Llama-3.2-1B-Instruct", choices=("meta-llama/Llama-3.2-3B-Instruct","meta-llama/Llama-3.2-1B-Instruct","meta-llama/Llama-3.2-1B","google/gemma-2-2b", "google/gemma-2b", "microsoft/Phi-3-mini-4k-instruct", "microsoft/phi-2", "gpt2-xl", "google/gemma-3-1b-pt"))
    parser.add_argument("--setting", type=str, default="lora", choices=("lora", "frozen", 'prefixtuning', "p_tuning", "prompttuning", "unfrozen"))
    parser.add_argument("--mapping_type", type=str, default="Transformer", choices=("MLPMixer","MLP", "Attention", "Transformer", "MOE"))
    parser.add_argument("--prefix_length", type=int, default=12)
    parser.add_argument("--clip_length", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--finetune_dataset_path", type=str, default="ptbxl", choices=('ptbxl','mimic'))
    parser.add_argument("--evaluate_dataset_path", type=str, default='ptbxl', choices=('ptbxl','mimic'))                        
    parser.add_argument("--question_type", type=str, default="all", choices=("all", "single-query", "single-verify", "single-choose"))
    parser.add_argument("--ecg_encoder_path", type=str, default="ckpts/Add_12Enc.pt")
    parser.add_argument("--load_from_mimic", type=str, default="")    
    parser.add_argument("--freeze_ecg_encoder", type=bool, default=False)
    parser.add_argument("--add_context", type=bool, default=True)
    parser.add_argument("--add_options", type=bool, default=True)
    parser.add_argument("--shuffle_options", type=bool, default=True)
    parser.add_argument("--lead_care", type=bool, default=False)
    parser.add_argument("--instruct_template", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--warmup_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters_to_accumulate", type=int, default=4)
    parser.add_argument("--logging_step", type=int, default=500)
    parser.add_argument("--validation_step", type=int, default=500)
    parser.add_argument("--out_dir", default="ckpts")
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--continue_training_from_checkpoint", type=str, default="") 

    args = parser.parse_args()
    print(vars(args))

    return args


if __name__ == "__main__":
    args = parse_argument() 
    print(f"Finetune LLM {args.setting}: {args.model_type}, Freeze ECG Encoder {args.ecg_encoder_path}: {args.freeze_ecg_encoder}, Hyper: LR-{args.lr}, Epochs-{args.epochs}, Batchsize-{args.batch_size} ...")

    if args.finetune_dataset_path == "mimic":
        finetune_dataset_path = "/workspace/ecg_foundation_model/data/processed_data"
        finetune_csv_base = "mimic_ecg_qa"
    elif args.finetune_dataset_path == "ptbxl":
        finetune_dataset_path = "/workspace/ecg_foundation_model/data/downstream/ptbxl/processed_qa_data"
        finetune_csv_base = "ptbxl_ecg_qa"
    else:
        raise "Only support MIMIC IV ECG and PTB-XL"  

    if args.evaluate_dataset_path == "mimic":
        evaluate_dataset_path = "/workspace/ecg_foundation_model/data/processed_data"
        evaluate_csv_base = "mimic_ecg_qa"
    elif args.evaluate_dataset_path == "ptbxl":
        evaluate_dataset_path = "/workspace/ecg_foundation_model/data/downstream/ptbxl/processed_qa_data"
        evaluate_csv_base = "ptbxl_ecg_qa"
    else:
        raise "Only support MIMIC IV ECG and PTB-XL"   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = CustomDataCollatorQA(model_type=args.model_type, add_context=args.add_context, instruct_template=args.instruct_template, add_options=args.add_options, shuffle_options=args.shuffle_options, lead_care=args.lead_care)

    ecg_encoder = get_model(args.ecg_encoder_path)

    train_data_csv_path = f"data/manifest/{finetune_csv_base}/train_qa_with_new_context.tsv"
    train_dataset = CustomECGQADataset(finetune_dataset_path, train_data_csv_path, question_type=args.question_type, args=args)

    val_data_csv_path = f"data/manifest/{finetune_csv_base}/valid_qa_with_new_context.tsv"
    val_dataset = CustomECGQADataset(finetune_dataset_path, val_data_csv_path, question_type=args.question_type, args=args)
    
    if args.freeze_ecg_encoder:
        for param in ecg_encoder.parameters():
            param.requires_grad = False
        ecg_encoder.eval()
            
    model = CustomECGQAModel(
        ecg_encoder=ecg_encoder,
        setting=args.setting,
        mapping_type=args.mapping_type,
        args=args,
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    if not args.eval:
        if args.load_from_mimic != "":
            model.load_state_dict(
                torch.load(args.load_from_mimic, map_location=torch.device("cuda"))
            )
            print(f"Loaded {args.load_from_mimic} Successfully from Pretrained MIMIC. Running evaluation on {evaluate_dataset_path}")

        model = custom_pytorch_model_run(train_dataset, val_dataset, collator, model, args, name=f"ckpts/PTBXL/v3/{args.model_type}-mapping_type-{args.mapping_type}-setting-{args.setting}-freeze_encoder-{args.freeze_ecg_encoder}-shuffle-{args.shuffle_options}-lead_care-{args.lead_care}-add_context-{args.add_context}-add_options-{args.add_options}-add_template-{args.instruct_template}_no_pos")
    else:
        print(args)
        set_random_seeds(args.seed) 
        checkpoint = "ckpts/PTBXL/v3/meta-llama/Llama-3.2-1B-Instruct-mapping_type-Transformer-setting-lora-freeze_encoder-False-shuffle-True-lead_care-False-add_context-True-add_options-True-add_template-True_48/best.bin"
        if args.verbose:
            print(f">> Loading pre-trained model {checkpoint}!")
        if os.path.exists(checkpoint):
            model.load_state_dict(
                torch.load(checkpoint, map_location=torch.device("cuda"))
            )
            print(f"Loaded {checkpoint} Successfully. Running evaluation on {evaluate_dataset_path}")

        else:
            raise ValueError("Please provide valid path for loading checkpoint")
        
        sample_data = None

        for qt in ["single-query","single-verify","single-choose"]:
            test_data_csv_path = f"data/manifest/{evaluate_csv_base}/test_qa_with_new_context.tsv" #data/manifest/ptbxl_ecg_qa/test_qa_10.tsv test_qa_with_new_context
            test_dataset = CustomECGQADataset(evaluate_dataset_path, test_data_csv_path, question_type=qt, sample_data=sample_data, args=args)
            custom_eval_llm_open_ended(model, test_dataset, args)
            print(f"Done evaluating {qt} over random {sample_data} samples in {evaluate_dataset_path} !!!")
    
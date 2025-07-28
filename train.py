import os
import copy
import time
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup, TrainingArguments, AutoTokenizer
from accelerate import Accelerator
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        ecg = inputs.pop('ecg')
        outputs = model(**inputs, ecg=ecg)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def custom_pytorch_model_run(train_dataset, val_dataset, data_collator, model, args=None, name=""):
    training_args = TrainingArguments(
        output_dir=name, 
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=args.validation_step,  
        save_steps=args.validation_step, 
        logging_steps=args.logging_step,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        save_total_limit=2,
        # weight_decay=0.01,
        warmup_ratio=0.1, 
        lr_scheduler_type="cosine",            
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        save_safetensors=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=data_collator.tokenizer
    )
    
    if args.continue_training_from_checkpoint != "":
        trainer.train(resume_from_checkpoint=args.continue_training_from_checkpoint) 
    else:
        trainer.train()


def pytorch_model_run(train_loader, valid_loader, model_obj, ignore_index, args):
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Using {device} ...")
    model = model_obj.to(device)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    elif args.load_pretrained:
        try:
            checkpoint = os.path.join(args.out_dir, "open_ended_latest.pt")
            if args.verbose:
                print(f">> Loading pre-trained model {checkpoint}!")
            if os.path.exists(checkpoint):
                model.load_state_dict(
                    torch.load(checkpoint, map_location=torch.device("cuda")), strict=False
                )
        except:
            print("Start Tuning ... ")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-6, betas=(0.9,0.98))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    valid_loader = accelerator.prepare(valid_loader)

    n_epochs = args.epochs
    accelerator.wait_for_everyone()
    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 

    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        for i, (prefix, tokens, mask, q_len) in tqdm(enumerate(train_loader)):
            with accelerator.accumulate(model):
                prefix = prefix.to(accelerator.device).float()  
                tokens = tokens.to(accelerator.device).long()
                mask = mask.to(accelerator.device).long()
                q_len = q_len.to(accelerator.device).long()
                outputs = model(prefix, tokens, mask, q_len)
                logits = outputs.logits

                loss = 0.0
                for b in range(logits.size(0)):
                    condensed_tokens = tokens[b,q_len[b]+model.prefix_length+1:] 
                    condensed_logits = logits[b,shift+q_len[b]+model.prefix_length:-1]
                    loss+= nnf.cross_entropy(
                        condensed_logits.reshape(-1,logits.shape[-1]), 
                        condensed_tokens.flatten(), 
                        ignore_index=ignore_index)
                
                loss = loss / logits.size(0)    
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                avg_loss = total_loss / (i+1)

        scheduler.step()

        torch.save(model.state_dict(), 
            os.path.join(args.out_dir, 
            f"open_ended_latest.pt"))
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, (prefix, tokens, mask, q_len) in tqdm(enumerate(valid_loader)):
                torch.cuda.empty_cache()

                prefix = prefix.to(accelerator.device).float()  
                tokens = tokens.to(accelerator.device).long()
                mask = mask.to(accelerator.device).long()
                q_len = q_len.to(accelerator.device).long()

                outputs = model(prefix, tokens, mask, q_len)
                logits = outputs.logits
                
                loss = 0.0
                for b in range(logits.size(0)):
                    condensed_tokens = tokens[b,q_len[b]+model.prefix_length+1:]
                    condensed_logits = logits[b,shift+q_len[b]+model.prefix_length:-1]
                    loss+= nnf.cross_entropy(
                        condensed_logits.reshape(-1,logits.shape[-1]), 
                        condensed_tokens.flatten(), 
                        ignore_index=ignore_index)
                loss=loss/logits.size(0)    
                total_loss += loss.item()
                avg_val_loss = total_loss / (i + 1)

        elapsed_time = time.time() - start_time
        print(
            "VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time
            )
        )

    return model

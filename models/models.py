from typing import Tuple, Optional, Union
import sys
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model,get_peft_config,PeftModelForCausalLM,TaskType,PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig 
from models.prefix_mappers import AttentionMapper, TransformerMapper, MoEMapper, MLPMixer
from models.ecg_encoder.ecg_encoder import get_ecg_feats


class CustomECGQAModel(nn.Module):
    def __init__(self, 
                ecg_encoder, 
                mapping_type="MLP",
                setting="lora",
                args=None,
                ):
        super(CustomECGQAModel, self).__init__()
        self.mapping_type = mapping_type
        self.setting = setting
        self.llm_type = args.model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_encoder = args.freeze_ecg_encoder
    
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        # self.llm = AutoModelForCausalLM.from_pretrained(self.llm_type,
        #     quantization_config=quantization_config, device_map="auto",
        #     torch_dtype="auto", trust_remote_code=True,  attn_implementation='eager')
        
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_type)#, device_map="auto", torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2")
        # self.llm = Gemma3ForCausalLM.from_pretrained(
        #     self.llm_type,
        #     device_map="auto"
        # )

        self.llm_embedding_size = self.llm.config.hidden_size
        if setting == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"] # NOTE
            )
            print("PEFT Setting: ", peft_config)
            self.llm = get_peft_model(self.llm, peft_config)

        elif setting == "frozen":
            for param in self.llm.parameters():
                param.requires_grad = False
            self.llm.eval()

        self.ecg_encoder = ecg_encoder
        self.postconv = nn.Conv1d(in_channels=312, out_channels=12, kernel_size=1)

        self.ecg_token_nums = args.prefix_length if mapping_type in ["Transformer","MLPMixer"] else 1
        self.ecg_feature_dim = 768
        
        if mapping_type == "MLP":  
            self.ecg_projection_layer = nn.Linear(self.ecg_feature_dim, self.llm_embedding_size)
        elif mapping_type == "MLPMixer":  
            self.ecg_projection_layer = MLPMixer(input_dim=self.ecg_feature_dim, llm_embedding_size=self.llm_embedding_size)
            
        elif mapping_type == "Attention":
            self.ecg_projection_layer = AttentionMapper(dim=self.ecg_feature_dim, output_dim=self.llm_embedding_size)
        elif mapping_type == "Transformer":
            self.ecg_projection_layer = TransformerMapper(
                dim_clip=self.ecg_feature_dim,
                dim_embedding=self.llm_embedding_size,
                prefix_length=self.ecg_token_nums,
                clip_length=args.clip_length,
                num_heads=4,
                num_layers=2
                )
            self.postlinear = nn.Linear(self.ecg_feature_dim, self.llm_embedding_size)

        elif mapping_type == "MOE":
            self.ecg_projection_layer = MoEMapper(self.ecg_feature_dim, self.llm_embedding_size)
        else:
            raise ValueError("Select valid mapping type: MLP, Attention and Transformer")

    def forward(self, input_ids=None, attention_mask=None, labels=None, ecg=None, prefix_len=None):
        ecg_features, conv_embedd = get_ecg_feats(self.ecg_encoder, ecg)
        ecg_features = ecg_features.reshape(input_ids.shape[0], 1, -1)

        if self.mapping_type != "Attention":
            if self.mapping_type == "MOE":
                embeddings = self.llm.get_input_embeddings()(input_ids)
                ecg_features_projected = self.ecg_projection_layer(ecg_features, embeddings)
                inputs_embeds = torch.cat((ecg_features_projected, embeddings), dim=1)
            else:
                ecg_features_projected = self.ecg_projection_layer(ecg_features)
                ########
                if self.mapping_type == "Transformer":
                    ecg_features_projected += self.postlinear(self.postconv(conv_embedd))
                ########
                embeddings = self.llm.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat((ecg_features_projected, embeddings), dim=1)
        elif self.mapping_type == "Attention":
            embeddings = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = self.ecg_projection_layer(ecg_features, embeddings, prefix_len)
        else:
            raise ValueError("Select valid mapping type: MLP, Attention and Transformer")

        if attention_mask is not None:
            attention_mask = torch.cat(
                (torch.ones((input_ids.size(0), self.ecg_token_nums), dtype=attention_mask.dtype).to(self.device), attention_mask),
                dim=1
            )
            
        if labels is not None:
            labels = torch.cat(
                (torch.full((labels.size(0), self.ecg_token_nums), -100, dtype=labels.dtype).to(self.device), labels),
                dim=1
            )
            
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

    def generate(self, input_ids=None, attention_mask=None, ecg=None, max_length=50, prefix_len=None, **kwargs):
        ecg_features, conv_embedd = get_ecg_feats(self.ecg_encoder, ecg.reshape(1,12,5000))
        ecg_features = ecg_features.reshape(input_ids.shape[0], 1, -1)

        if self.mapping_type != "Attention":
            if self.mapping_type == "MOE":
                embeddings = self.llm.get_input_embeddings()(input_ids)
                ecg_features_projected = self.ecg_projection_layer(ecg_features, embeddings)
                inputs_embeds = torch.cat((ecg_features_projected, embeddings), dim=1)
            else:
                ecg_features_projected = self.ecg_projection_layer(ecg_features)
                if self.mapping_type == "Transformer":
                    ecg_features_projected += self.postlinear(self.postconv(conv_embedd))
                embeddings = self.llm.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat((ecg_features_projected, embeddings), dim=1)
        elif self.mapping_type == "Attention":
            embeddings = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = self.ecg_projection_layer(ecg_features, embeddings, prefix_len)
        else:
            raise ValueError("Select valid mapping type: MLP, Attention and Transformer")

        if attention_mask is not None:
            attention_mask = torch.cat(
                (torch.ones((input_ids.size(0), self.ecg_token_nums), dtype=attention_mask.dtype).to(self.device), attention_mask),
                dim=1
            )
        generated_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=inputs_embeds.shape[1] + max_length,
            attention_mask=attention_mask,
            **kwargs
        )
        return generated_outputs

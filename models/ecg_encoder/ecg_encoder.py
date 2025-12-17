from copy import deepcopy
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
import sys
from models.ecg_encoder.dbeta import M3AEModel
from types import SimpleNamespace


def get_model(checkpoint_path="") :
    with open('configs/ECGEncoder.json', 'r') as json_file:
        cfg = json.load(json_file)

    cfg = SimpleNamespace(**cfg['model'])
    model = M3AEModel(cfg)
    checkpoint = torch.load(checkpoint_path)
    if "ecg_encoder.mask_emb" in checkpoint["model"].keys():
        del checkpoint["model"]["ecg_encoder.mask_emb"]

    model.load_state_dict(checkpoint["model"], strict=True)
    model.remove_pretraining_modules()
    print("Loaded ECG encoder ...")
    
    return model


def get_ecg_feats(model, ecgs, args=None):        

    uni_modal_ecg_feats, ecg_padding_mask, conv_embedd = (
        model.ecg_encoder.get_embeddings(ecgs, padding_mask=None)
    )
    
    cls_emb = model.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
    uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)
    uni_modal_ecg_feats = model.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
    out = model.multi_modal_ecg_proj(uni_modal_ecg_feats)
    ecg_features = model.unimodal_ecg_pooler(out)
        
    return ecg_features, conv_embedd


if __name__ == "__main__":
    model = get_model()
    ecgs = torch.randn(4, 12, 5000, requires_grad=True)
    ecg_features = get_ecg_feats(model, ecgs)  
    print(ecg_features.shape) 
    
    
    
    
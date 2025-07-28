import math
from collections import OrderedDict
import logging

import numpy as np
import torch
import torch.nn as nn

from transformers.models.t5.modeling_t5 import (
    T5EncoderModel
)

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPredictionHeadTransform
)

from .cross_layer import BertCrossLayer
from .base import PretrainingModel
from .transformer import ECGTransformerModel

import warnings
warnings.filterwarnings("ignore")
import urllib3
urllib3.disable_warnings()
logger = logging.getLogger(__name__)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class M3AEModel(PretrainingModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

        self.vocab_size = cfg.vocab_size

        self.mim_prob = cfg.mim_prob
        self.mim_layer = cfg.mim_layer
        

        self.ecg_encoder = ECGTransformerModel(cfg)
        
        self.class_embedding = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.language_encoder = T5EncoderModel.from_pretrained("google/flan-t5-base") 

        self.language_encoder.pooler = None

        self.multi_modal_language_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim) 
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_ecg_proj = nn.Linear(cfg.encoder_embed_dim, cfg.hidden_dim)
        self.multi_modal_ecg_proj.apply(init_weights)
        
        self.modality_type_embeddings = nn.Embedding(2, cfg.hidden_dim)
        self.modality_type_embeddings.apply(init_weights)
        
        bert_config = BertConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_dim,
            num_hidden_layers=cfg.num_layers,
            num_attention_heads=cfg.num_heads,
            intermediate_size=cfg.hidden_dim * 4,
            max_position_embeddings=cfg.max_text_size,
            hidden_dropout_prob=cfg.drop_rate,
            attention_probs_dropout_prob=cfg.drop_rate
        ) 
        self.multi_modal_ecg_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)]
        )
        self.multi_modal_ecg_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(cfg.num_top_layer)]
        )
        self.multi_modal_language_layers.apply(init_weights)
    
        self.multi_modal_ecg_pooler = Pooler(cfg.hidden_dim)
        self.multi_modal_ecg_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(cfg.hidden_dim)
        self.multi_modal_language_pooler.apply(init_weights)
        
        ## NOTE SIGLEP ##
    
        self.unimodal_ecg_pooler = Pooler(cfg.hidden_dim)
        self.unimodal_ecg_pooler.apply(init_weights)
        self.unimodal_language_pooler = Pooler(cfg.hidden_dim)
        self.unimodal_language_pooler.apply(init_weights)  

        ## ### ##

        self.mlm_head = MLMHead(bert_config)
        self.mlm_head.apply(init_weights)
        self.mim_head = MIMHead(cfg)
        self.mim_head.apply(init_weights)
        self.itm_head = ITMHead(cfg.hidden_dim * 2)
        self.itm_head.apply(init_weights)
        
    @classmethod
    def build_model(cls, cfg, task=None):
        """Build a new model instance."""
        return cls(cfg)

    def random_masking(self, x, mask_ratio):
        x_ = x[:, :1]
        x = x[:, 1:]
        
        bsz, tsz, csz = x.shape
        len_keep = int(tsz * (1 - mask_ratio))
        
        noise = torch.rand(bsz, tsz, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, csz))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bsz, tsz], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # append cls token
        x_masked = torch.cat((x_, x_masked), dim=1)
        
        return x_masked, mask, ids_restore

    def forward(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_attention_mask,
        mask=True,
        features_only=False,
        **kwargs
    ):
        if ecg_padding_mask is not None and not ecg_padding_mask.any():
            ecg_padding_mask = None

        assert ecg_padding_mask is None, (
            "all the ecgs in a batch should have the same size for M3AE model."
        )

        ret = dict()
        text_input_shape = text_attention_mask.size()
        extended_text_masks = self.language_encoder.get_extended_attention_mask(text_attention_mask, text_input_shape)
        uni_modal_text_feats = self.language_encoder(input_ids=text)[0] 

        uni_modal_text_feats = self.multi_modal_language_proj(uni_modal_text_feats)
        
        # NOTE This is for SIGLEP #
        ret["uni_modal_text_feats"] = self.unimodal_language_pooler(uni_modal_text_feats)
        
        uni_modal_ecg_feats, ecg_padding_mask, conv_embedd = (
            self.ecg_encoder.get_embeddings(ecg, padding_mask=ecg_padding_mask)
        )
                
        cls_emb = self.class_embedding.repeat((len(uni_modal_ecg_feats), 1, 1))
        self.cls_emb = cls_emb
        uni_modal_ecg_feats = torch.cat([cls_emb, uni_modal_ecg_feats], dim=1)

        if mask:
            uni_modal_ecg_feats, mim_masks, mim_ids_restore = self.random_masking(
                uni_modal_ecg_feats, self.mim_prob
            )
            uni_modal_ecg_feats = self.ecg_encoder.get_output(uni_modal_ecg_feats)
            ret["mim_masks"] = mim_masks
            ret["mim_ids_restore"] = mim_ids_restore
        else:
            uni_modal_ecg_feats = self.ecg_encoder.get_output(uni_modal_ecg_feats, ecg_padding_mask)
        
        uni_modal_ecg_feats = self.multi_modal_ecg_proj(uni_modal_ecg_feats)

        # NOTE This is for SIGLEP #
        ret["uni_modal_ecg_feats"] = self.unimodal_ecg_pooler(uni_modal_ecg_feats)

        if ecg_padding_mask is not None and ecg_padding_mask.any():
            ecg_attention_mask = ~ecg_padding_mask
        else:
            ecg_attention_mask = torch.ones(
                (uni_modal_ecg_feats.size(0), uni_modal_ecg_feats.size(1)),
                dtype=torch.long,
                device=ecg.device
            )
        extended_ecg_masks = (
            self.language_encoder.get_extended_attention_mask(ecg_attention_mask, ecg_attention_mask.size())
        )

        uni_modal_text_feats, uni_modal_ecg_feats = (
            uni_modal_text_feats + self.modality_type_embeddings(torch.zeros_like(text, dtype=int)),
            uni_modal_ecg_feats + self.modality_type_embeddings(torch.ones_like(ecg_attention_mask, dtype=int))
        )

        x, y = uni_modal_text_feats, uni_modal_ecg_feats
        for layer_idx, (text_layer, ecg_layer) in enumerate(
            zip(self.multi_modal_language_layers, self.multi_modal_ecg_layers)
        ):
            if mask and self.mim_layer == layer_idx:
                (
                    ret[f"multi_modal_text_feats_{layer_idx}"],
                    ret[f"multi_modal_ecg_feats_{layer_idx}"]
                ) = x, y
            
            x1 = text_layer(x, y, extended_text_masks, extended_ecg_masks, output_attentions=True)
            y1 = ecg_layer(y, x, extended_ecg_masks, extended_text_masks, output_attentions=True)
            x, y = x1[0], y1[0]
        
        multi_modal_text_feats, multi_modal_ecg_feats = x, y
        multi_modal_text_cls_feats = self.multi_modal_language_pooler(x)
        multi_modal_ecg_cls_feats = self.multi_modal_ecg_pooler(y)
        multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_ecg_cls_feats], dim=-1)
        
        ret.update({
            "multi_modal_text_feats": multi_modal_text_feats,
            "multi_modal_ecg_feats": multi_modal_ecg_feats,
            "multi_modal_cls_feats": multi_modal_cls_feats,
            "conv_embedd": conv_embedd
        })

        if features_only:
            return ret

        ret["ecgs"] = ecg
        if mask:
            mlm_logits = self.mlm_head(multi_modal_text_feats)
            if self.mim_layer == -1:
                mim_logits = self.mim_head(multi_modal_ecg_feats, mim_ids_restore)
            else:
                mim_logits = (
                    self.mim_head(ret[f"multi_modal_ecg_feats_{self.mim_layer}"], mim_ids_restore)
                )
            
            itm_logits = self.itm_head(multi_modal_cls_feats)

            ret.update({
                "mlm_logits": mlm_logits,
                "mim_logits": mim_logits,
                "itm_logits": itm_logits
            })
        else: 
            itm_logits = self.itm_head(multi_modal_cls_feats)
            ret.update({
                "itm_logits": itm_logits
            })
        
        return ret

    def get_logits(self, net_output):
        bsz, tsz, _, _ = net_output["mim_logits"].shape
        res = {
            "itc_uni_modal_feats": (net_output["uni_modal_ecg_feats"], net_output["uni_modal_text_feats"]), 
            "mlm_logits": net_output["mlm_logits"].view(-1, self.vocab_size),
            "mim_logits": net_output["mim_logits"].view(bsz, tsz, -1),
            "itm_logits": net_output["itm_logits"],
        }
        
        return res

    def get_targets(self, sample, net_output, norm_pix_loss=True):
        mlm_target = sample["mlm_labels"].view(-1)

        mim_target = sample["net_input"]["ecg"]

        mim_logits = net_output["mim_logits"]
        if mim_target.size(-1) > mim_logits.size(1) * mim_logits.size(-1):
            offset = mim_target.size(-1) - (mim_logits.size(1) * mim_logits.size(-1))
            mim_target = mim_target[:, :, :-offset]

        if norm_pix_loss:
            mean = mim_target.mean(dim=-1, keepdim=True)
            var = mim_target.var(dim=-1, keepdim=True)
            mim_target = (mim_target - mean) / (var + 1.e-6) ** .5
        num_patches = mim_logits.size(1)
        mim_target = mim_target.view(mim_target.size(0), mim_target.size(1), num_patches, -1)
        mim_target = mim_target.permute(0, 2, 1, 3)
        mim_target = mim_target.contiguous().view(mim_target.size(0), mim_target.size(1), -1)

        itm_target = sample["is_aligned"]
        
        return {
            "mlm_target": mlm_target,
            "mim_target": mim_target,
            "itm_target": itm_target.long()
        }

    def extract_features(
        self,
        ecg,
        text,
        ecg_padding_mask,
        text_attention_mask,
        ecg_2,
        ecg_2_padding_mask,
        mask
    ):
        res = self.forward(
            ecg=ecg,
            text=text,
            ecg_padding_mask=ecg_padding_mask,
            text_attention_mask=text_attention_mask,
            ecg_2=ecg_2,
            ecg_2_padding_mask=ecg_2_padding_mask,
            mask=mask,
            features_only=True
        )
        return res

    def remove_pretraining_modules(self):
        self.mlm_head = None
        self.mim_head = None
        self.itm_head = None
        self.language_encoder = None
        self.multi_modal_language_layers = None
        self.multi_modal_ecg_layers = None
                

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config) 
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class MIMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.decoder_hidden_dim = cfg.mim_decoder_hidden_dim
        self.decoder_num_layers = cfg.mim_decoder_num_layers
        self.decoder_num_heads = cfg.mim_decoder_num_heads

        self.decoder_embed = nn.Linear(self.hidden_dim, self.decoder_hidden_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = PositionalEncoding(self.decoder_hidden_dim, max_len=512)

        self.decoder = Transformer(self.decoder_hidden_dim, self.decoder_num_layers + 1, self.decoder_num_heads)
        self.decoder_norm = LayerNorm(self.decoder_hidden_dim)

        def _conv_out_length(input_length, kernel_size, stride):
            return np.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(cfg.conv_feature_layers)

        dummy_input_length = 5000
        inferred_input_length = dummy_input_length
        for i in range(len(conv_cfg_list)):
            inferred_input_length = _conv_out_length(inferred_input_length, conv_cfg_list[i][1], conv_cfg_list[i][2])
        self.inferred_decoded_size = int(np.floor(dummy_input_length / inferred_input_length))

        self.decoder_pred = nn.Linear(self.decoder_hidden_dim, self.inferred_decoded_size * 12, bias=True)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = self.decoder_pos_embed(x)

        # apply Transformer blocks
        x = x.permute(1, 0, 2)  # (bsz, tsz, csz) -> (tsz, bsz, csz)
        x = self.decoder(x)
        x = x.permute(1, 0, 2)  # (tsz, bsz, csz) -> (bsz, tsz, csz)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        x = x.view(x.size(0), x.size(1), -1, self.inferred_decoded_size)

        return x

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x): 
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, :x.size(1)]
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x_mask: torch.Tensor):
        if x_mask is not None:
            x_mask = x_mask.to(dtype=torch.bool, device=x.device)
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=x_mask)[0]

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), x_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers - 1)])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, x_mask)
        return x
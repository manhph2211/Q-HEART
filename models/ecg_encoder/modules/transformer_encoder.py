import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.ecg_encoder.modules import (
    TransformerEncoderLayer,
    SwitchTransformerEncoderLayer,
    MultiHeadAttention,
    LayerNorm
)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiHeadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embed_dim = args.encoder_embed_dim
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer( # NOTE change to SwitchTransformerEncoderLayer from TransformerEncoderLayer
                    embed_dim=self.embed_dim,
                    ffn_dim=args.encoder_ffn_embed_dim,
                    n_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    layer_norm_first=args.layer_norm_first,
                ) for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embed_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(
        self,
        x,
        padding_mask=None,
        attn_mask=None,
    ):
        x = self.extract_features(x, padding_mask, attn_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(
        self,
        x,
        padding_mask=None,
        attn_mask=None
    ):
        if padding_mask is not None:
            x[padding_mask] = 0

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    self_attn_mask=attn_mask,
                    need_weights=False
                )
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0,1)

        return x
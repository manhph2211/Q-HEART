# source: https://github.com/rmokady/CLIP_prefix_caption
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple, Optional, Union
from torch import einsum
from einops import rearrange

import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 4, num_heads: int = 4):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, num_heads, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)
        
        
class AttentionMapper(nn.Module):
    def __init__(self, dim=786, output_dim=2048, num_heads=8, dim_head=64):
        super(AttentionMapper, self).__init__()
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        
        # ECG projection layer
        self.ecg_projection_layer = nn.Linear(dim, output_dim)

        # Normalization layers
        self.norm_ecg = nn.LayerNorm(output_dim)
        self.norm_query = nn.LayerNorm(output_dim)

        # Attention projections
        self.to_q = nn.Linear(output_dim, self.inner_dim, bias=False)  # Queries from query_features
        self.to_kv = nn.Linear(output_dim, self.inner_dim * 2, bias=False)  # Keys and values from ecg_features

        # Output projection
        self.to_out = nn.Linear(self.inner_dim, output_dim, bias=False)

    def forward(self, ecg_features, query_features, prefix_len=None):
        """
        Forward pass for AttentionMapper.
        
        Args:
            ecg_features: Tensor of shape (batch_size, num_ecg_tokens, dim).
            query_features: Tensor of shape (batch_size, seq_len, output_dim).
        
        Returns:
            Concatenated tensor of shape (batch_size, seq_len + num_ecg_tokens, output_dim).
        """
        # Project and normalize ECG features
        ecg_features = self.ecg_projection_layer(ecg_features)  # (B, num_ecg_tokens, output_dim)
        ecg_features = self.norm_ecg(ecg_features)  # Normalize ECG features

        # Normalize query features
        if prefix_len is not None:
            normed_query_features = self.norm_query(query_features[:,:prefix_len,:]) # (B, seq_len, output_dim)
        else:
            normed_query_features = self.norm_query(query_features)

        # Compute Q (from query_features), K and V (from ecg_features)
        q = self.to_q(ecg_features)  # (B, seq_len, inner_dim)
        k, v = self.to_kv(normed_query_features).chunk(2, dim=-1)  # Split ECG features into keys and values

        # Reshape for multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q, k, v)
        )  # (B, num_heads, seq_len or num_ecg_tokens, dim_head)

        # Compute scaled dot-product attention
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) * self.scale  # (B, num_heads, seq_len, num_ecg_tokens)
        attn = scores.softmax(dim=-1)  # Attention weights

        # Apply attention weights to ECG values
        out = torch.einsum("bhqk, bhvd -> bhqd", attn, v)  # (B, num_heads, seq_len, dim_head)

        # Reshape back to (B, seq_len, output_dim)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # Concatenate the attended ECG features with the query features
        return torch.cat((out, query_features), dim=1)  # (B, seq_len + num_ecg_tokens, output_dim)


class MoEMapper(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=12):
        super(MoEMapper, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
            ) for _ in range(num_experts)
        ])
        self.text_gate = nn.Linear(output_dim, num_experts)
        self.output_dim = output_dim

    def forward(self, x, t):
        B = x.size(0)
        x_flat = x.squeeze(1)            # [B, input_dim]
        t_mean = t.mean(dim=1)           # [B, output_dim]

        gate_logits = self.text_gate(t_mean) #+ self.x_gate(x_flat)  # [B, num_experts]
        top1_indices = gate_logits.argmax(dim=-1)                   # [B]

        moe_out = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)

        for expert_id in range(self.num_experts):
            mask = (top1_indices == expert_id)
            if mask.any():
                moe_out[mask] = self.experts[expert_id](x_flat[mask])

        return moe_out.reshape(B, 1, self.output_dim) 


class MultiLayerPerceptron(nn.Module):
    def __init__(self, hidden_size, depth):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            *[
                nn.Sequential(nn.GELU(), nn.Linear(hidden_size, hidden_size))
                for _ in range(depth - 1)
            ]
        )

    def forward(self, x):
        return self.mlp(x)


class MultiModalProjector(nn.Module):
    def __init__(self, input_size, output_size, mlp_depth, proj_out_num=256):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.mm_projector = nn.Sequential(
            nn.Linear(input_size, output_size),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(output_size, output_size),
                )
                for _ in range(mlp_depth - 1)
            ]
        )

    def forward(self, x):
        return self.mm_projector(x)


class LowHighHybridMLP(nn.Module):
    def __init__(
        self, low_input_size, high_input_size, output_size, mlp_depth, proj_out_num=288
    ):
        super().__init__()
        self.proj_out_num = proj_out_num
        self.low_up_mlp = nn.Linear(low_input_size, output_size)
        self.high_up_mlp = nn.Linear(high_input_size, output_size)
        modules = []
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_size, output_size))
        self.mm_projector = nn.Sequential(*modules)

    def forward(self, x):
        low_x, high_x = x

        low_x = self.low_up_mlp(low_x)
        high_x = self.high_up_mlp(high_x)
        x = torch.cat([low_x, high_x], dim=1)

        x = self.mm_projector(x)

        return x


class MixerLayer(nn.Module):
    def __init__(self, input_size, output_size, mlp_depth=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_size[1])
        self.ln2 = nn.LayerNorm(input_size[1])

        self.mlp1 = MultiModalProjector(
            input_size=input_size[0], output_size=output_size[0], mlp_depth=mlp_depth
        )
        self.mlp2 = MultiModalProjector(
            input_size=input_size[1], output_size=output_size[1], mlp_depth=mlp_depth
        )

    def forward(self, x):
        x = self.ln1(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.mlp1(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.ln2(x)
        x = self.mlp2(x)

        return x


class MixerLowHighHybridMLP(nn.Module):
    def __init__(
        self,
        low_input_size: tuple = (256, 384),
        low_output_size: list = [192, 128],
        high_input_size: tuple = (32, 768),
        high_output_size: list = [64, 128],
        output_dim=3584,
        depth=2,
        mlp_depth=2,
        proj_out_num=256,
    ):
        assert (
            len(low_output_size) == len(high_output_size) == depth
        ), "Output size must be same for both low and high input"
        assert output_dim % (2**depth) == 0, "Output dim must be divisible by 2**depth"

        super().__init__()

        self.proj_out_num = proj_out_num

        self.low_mixer = nn.ModuleList(
            [
                MixerLayer(
                    input_size=(
                        (low_output_size[i - 1], output_dim // (2 ** (depth - i)))
                        if i > 0
                        else low_input_size
                    ),
                    output_size=(
                        low_output_size[i],
                        output_dim // (2 ** (depth - i - 1)),
                    ),
                    mlp_depth=mlp_depth,
                )
                for i in range(depth)
            ]
        )
        self.high_mixer = nn.ModuleList(
            [
                MixerLayer(
                    input_size=(
                        (high_output_size[i - 1], output_dim // (2 ** (depth - i)))
                        if i > 0
                        else high_input_size
                    ),
                    output_size=(
                        high_output_size[i],
                        output_dim // (2 ** (depth - i - 1)),
                    ),
                    mlp_depth=mlp_depth,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        low_x, high_x = x
        for low_layer, high_layer in zip(self.low_mixer, self.high_mixer):
            low_x = low_layer(low_x)
            high_x = high_layer(high_x)
        x = torch.cat([low_x, high_x], dim=1)

        return x
    

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MLPMixer(nn.Module):
    def __init__(self, num_tokens=12, input_dim=768, llm_embedding_size=1024, num_layers=4):
        super().__init__()
        self.token_mixer = nn.ModuleList([
            MLPBlock(num_tokens, input_dim) for _ in range(num_layers)
        ])
        self.channel_mixer = nn.ModuleList([
            MLPBlock(input_dim, input_dim) for _ in range(num_layers)
        ])
        self.last = nn.Linear(input_dim, llm_embedding_size)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        for token_mlp, channel_mlp in zip(self.token_mixer, self.channel_mixer):
            x = x + token_mlp(x.transpose(1, 2)).transpose(1, 2)  
            x = x + channel_mlp(x)  
        return self.last(self.norm(x))
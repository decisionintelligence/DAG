import numpy as np
import torch
import torch.nn as nn
from math import sqrt

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask,
        exog_attn=None,
        attn_alpha=0.5,
        tau=None,
        delta=None,
        channel_mask=None,
        residual_beta=1.0,
    ):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.mask, -np.inf)
        elif self.mask_flag and attn_mask is None:
            causal_mask = TriangularCausalMask(B, L, device=queries.device)
            scores = scores.masked_fill(causal_mask.mask, -np.inf)

        fused_scores = scale * scores
        if exog_attn is not None:
            # SWAN v2: 在logit空间融合外生相关先验，避免概率空间再softmax带来的数值失真
            exog_logits = torch.log(torch.clamp(exog_attn, min=1e-8))
            fused_scores = fused_scores + attn_alpha * exog_logits

        full_attn = torch.softmax(fused_scores, dim=-1)

        if channel_mask is not None:
            # SWAN v2: 先mask logits再softmax，最后与full attention凸组合，保证梯度稳定
            masked_scores = fused_scores.masked_fill(channel_mask == 0, -1e9)
            masked_attn = torch.softmax(masked_scores, dim=-1)
            attn_before_dropout = residual_beta * masked_attn + (1 - residual_beta) * full_attn
        else:
            attn_before_dropout = full_attn

        attn = self.dropout(attn_before_dropout)
        output = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return output.contiguous(), attn_before_dropout
        return output.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask,
        exog_attn=None,
        attn_alpha=0.5,
        tau=None,
        delta=None,
        channel_mask=None,
        residual_beta=1.0,
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            exog_attn=exog_attn,
            attn_alpha=attn_alpha,
            tau=tau,
            delta=delta,
            channel_mask=channel_mask,
            residual_beta=residual_beta,
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

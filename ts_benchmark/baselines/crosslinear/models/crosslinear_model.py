import math
import torch
import torch.nn as nn
from einops import rearrange

from ..layers.SelfAttention_Family import FullAttention, AttentionLayer


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU())

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        # 在第2个维度，随后进行维度映射，不同论文似乎方法略有不同
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len))

    def forward(self, x):
        # (B,1,N,d) -> (B,1,N*d) -> (B,1，d_ff) -> (B,1,pred_len)
        x = self.linear(x)
        return x


class CrossLinear_model(nn.Module):
    def __init__(self, configs):
        super(CrossLinear_model, self).__init__()
        self.task_name = configs.task_name
        self.ms = configs.series_dim == 1
        self.EPS = 1e-5
        patch_len = configs.patch_len
        patch_num = math.ceil(configs.seq_len / patch_len)
        variate_num = 1 if self.ms else configs.dec_in
        # embedding
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha)
        self.beta = nn.Parameter(torch.ones([1]) * configs.beta)
        self.correlation_embedding = nn.Conv1d(configs.dec_in, variate_num, 3, padding='same')
        self.value_embedding = Patch_Embedding(configs.seq_len, patch_num, patch_len, configs.d_model, configs.d_ff, variate_num)
        self.pos_embedding = nn.Parameter(torch.randn(1, variate_num, patch_num, configs.d_model))
        # head
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = De_Patch_Embedding(
                configs.pred_len, patch_num, configs.d_model, configs.d_ff, variate_num)


    def forecast(self, x_enc):
        # 将第一个 channel 移到最后
        x_enc = torch.cat([x_enc[:, :, 1:], x_enc[:, :, 0:1]], dim=-1)
        # B,C,L
        x_enc = x_enc.permute(0, 2, 1)


        # normalization
        # 取出特征 B,1,L ->B,L
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (torch.std(x_enc, dim=-1, keepdim=True) + self.EPS)
        # embedding
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        # object先基于correlation_embedding聚合的特征编码，这里其实更像是集中考虑了全部的特征，并和目标再一次进行了融合
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(x_enc)

        # 并按照patch粒度进行了嵌入
        # B,1,N,d  pos nn.parameter B,1,N,d
        x_obj = self.beta * self.value_embedding(x_obj) + (1 - self.beta) * self.pos_embedding
        # head
        # 转为B，1，pred
        y_out = self.head(x_obj)
        # de-normalization
        y_out = y_out * std + mean
        y_out = y_out.permute(0, 2, 1)
        return y_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out
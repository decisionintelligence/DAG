import math

import torch
from torch import nn

from ts_benchmark.baselines.GCGNet.layers.Embed import PatchEmbedding
from ts_benchmark.baselines.GCGNet.layers.blocks import VAE, Sparsifier
from ts_benchmark.baselines.GCGNet.layers.graph import GraphDiscriminator, GCNStack


class FlattenHead(nn.Module):
    def __init__(self, n_vars, endo_num, nf, d_ff, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(n_vars, endo_num)
        self.mlp = nn.Sequential(
            nn.Linear(nf, d_ff),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_ff, target_window),
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, nvars, patch_num, d_model]
        x = x.permute(0, 3, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 3, 2, 1)
        x = self.flatten(x)
        x = self.mlp(x)
        x = self.dropout(x)
        return x


class GCGNetModel(nn.Module):
    def __init__(self, config):
        super(GCGNetModel, self).__init__()
        self.var_num = config.enc_in
        self.endo_num = config.series_dim
        self.exog_num = config.enc_in - config.series_dim

        self.patch_len = config.patch_len
        self.input_patch_num = config.seq_len // config.patch_len
        self.input_len = config.patch_len * self.input_patch_num

        self.pred_len = config.pred_len
        self.pred_patch_num = math.ceil(config.pred_len / config.patch_len)

        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.rank = config.rank
        self.e_layers = config.e_layers
        self.n_heads = config.n_heads

        self.dropout = config.dropout
        self.use_norm = config.use_norm
        self.use_future_exog = config.use_future_exog

        self.graph_criterion = nn.L1Loss()

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.patch_len,
            dropout=self.dropout
        )
        self.vae = VAE(
            input_len=self.d_model * self.input_patch_num,
            output_len=self.pred_len,
            d_model=self.d_model,
            d_ff=self.d_ff,
        )

        self.graph_discriminator = GraphDiscriminator(
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            nodes_num=self.var_num * (self.input_patch_num + self.pred_patch_num),
            rank=self.rank
        )

        self.sparsifier = Sparsifier(
            n_vars=self.var_num,
            patch_num=self.var_num * (self.input_patch_num + self.pred_patch_num),
        )

        self.gcn = GCNStack(
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
        )

        self.head = FlattenHead(
            n_vars=self.var_num,
            endo_num=self.endo_num,
            d_ff=self.d_ff,
            nf=self.d_model * (self.input_patch_num + self.pred_patch_num),
            target_window=self.pred_len,
            head_dropout=self.dropout
        )

        self.graph_loss_list = []

    def forward(self, history, exog_future, endo_future=None):
        endo_history = history[:, -self.input_len:, :self.endo_num]
        exog_history = history[:, -self.input_len:, self.endo_num:]
        exog_future = exog_future[:, :self.pred_len, :]
        endo_future = endo_future[:, :self.pred_len, :]
        if self.training:
            assert endo_future is not None, "endo_future must be provided during training"

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            endo_means = endo_history.mean(1, keepdim=True).detach()
            endo_stdev = torch.sqrt(
                torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            if self.use_future_exog:
                exog = torch.cat([
                    exog_history,
                    exog_future
                ], dim=1)
            else:
                exog = exog_history
            exog_means = exog.mean(1, keepdim=True).detach()
            exog_stdev = torch.sqrt(
                torch.var(exog, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

            endo_history = self.sample_norm(endo_history, endo_means, endo_stdev)
            if self.training:
                endo_future = self.sample_norm(endo_future, endo_means, endo_stdev)

            exog_history = self.sample_norm(exog_history, exog_means, exog_stdev)
            exog_future = self.sample_norm(exog_future, exog_means, exog_stdev)

        history = torch.cat([endo_history, exog_history], dim=-1)[:, -self.input_len:, :]  # [batch, input_len, var_num]
        history = history.permute(0, 2, 1)  # [batch, var_num, input_len]

        patch_history, _ = self.patch_embedding(history)
        patch_history = patch_history.reshape(
            -1, self.var_num, self.input_patch_num * self.d_model
        )
        predict_future, mu, logvar = self.vae(patch_history)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        predict_endo_future = predict_future[:, :self.endo_num, :].permute(0, 2, 1)

        if self.use_future_exog:
            mix_future = torch.cat([predict_endo_future, exog_future], dim=-1).permute(0, 2, 1)
            predict_full_sequence = torch.cat([history, mix_future], dim=-1)
        else:
            predict_full_sequence = torch.cat([history, predict_future], dim=-1)

        patch_predict_full_sequence, _ = self.patch_embedding(predict_full_sequence)
        patch_predict_full_sequence = patch_predict_full_sequence.reshape(
            -1, self.var_num * (self.input_patch_num + self.pred_patch_num), self.d_model
        )
        predict_adj, mu, logvar = self.graph_discriminator(patch_predict_full_sequence)
        graph_kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        graph_loss = 0
        if self.training:
            target_future = torch.cat([endo_future, exog_future], dim=-1).permute(0, 2, 1)
            target_full_sequence = torch.cat([history, target_future], dim=-1)

            patch_target_full_sequence, _ = self.patch_embedding(target_full_sequence)
            patch_target_full_sequence = patch_target_full_sequence.reshape(
                -1, self.var_num * (self.input_patch_num + self.pred_patch_num), self.d_model
            )
            target_adj, _, _ = self.graph_discriminator(patch_target_full_sequence)
            graph_loss = self.graph_criterion(predict_adj, target_adj.detach())

        predict_adj, moe_loss = self.sparsifier(predict_adj, is_training=self.training)
        patch_full_enc = self.gcn(predict_adj, patch_predict_full_sequence)
        patch_full_enc = patch_full_enc.reshape(
            -1, self.var_num, self.input_patch_num + self.pred_patch_num, self.d_model
        )
        patch_endo_enc = patch_full_enc[:, :, :, :]

        endo_target_dec = self.head(patch_endo_enc).permute(0, 2, 1)
        if self.use_norm:
            endo_target_dec = self.sample_denorm(endo_target_dec, endo_means, endo_stdev)

        return endo_target_dec, graph_loss + moe_loss + kl_loss + graph_kl_loss

    def sample_norm(self, x, means, stdev):
        x = x - means
        x /= stdev
        return x

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x

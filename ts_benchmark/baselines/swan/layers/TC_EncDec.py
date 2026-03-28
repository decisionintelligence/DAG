import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.swan.layers.Embed import PatchEmbedding, CompressAndProject
from ts_benchmark.baselines.swan.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.swan.layers.Transformer_EncDec import Encoder, EncoderLayer


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ExogenousWeightModule(nn.Module):
    def __init__(
        self,
        exog_dim,
        series_dim,
        d_model,
        lambda3,
        warmup_steps,
        weight_threshold,
        tau_init,
        tau_min,
        tau_decay,
    ):
        super(ExogenousWeightModule, self).__init__()
        self.exog_dim = exog_dim
        self.series_dim = series_dim
        self.lambda3 = lambda3
        self.warmup_steps = max(1, warmup_steps)
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.exog_to_series = nn.Linear(exog_dim, series_dim)
        self.threshold = nn.Parameter(torch.tensor(weight_threshold, dtype=torch.float32))
        self.register_buffer("tau", torch.tensor(tau_init, dtype=torch.float32))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def _update_tau(self):
        if self.training:
            self.step += 1
            new_tau = torch.clamp(self.tau * self.tau_decay, min=self.tau_min)
            self.tau = new_tau

    def forward(self, patch_exog, batch_size, exog_vars):
        patch_num = patch_exog.shape[1]
        d_model = patch_exog.shape[2]
        reshaped = patch_exog.view(batch_size, exog_vars, patch_num, d_model)
        pooled = reshaped.mean(dim=2)
        raw_weight = torch.sigmoid(self.weight_mlp(pooled)).squeeze(-1)
        self._update_tau()
        tau = self.tau.detach()
        smooth_weight = torch.sigmoid((raw_weight - self.threshold) / (tau + 1e-6))
        soft_weight = F.relu(smooth_weight - self.threshold)
        weighted_patch = reshaped * soft_weight.unsqueeze(-1).unsqueeze(-1)
        weighted_patch = weighted_patch.view(batch_size * exog_vars, patch_num, d_model)
        ramp = min(1.0, float(self.step.item()) / float(self.warmup_steps))
        sparse_loss = self.lambda3 * ramp * soft_weight.mean()
        series_gate = torch.sigmoid(self.exog_to_series(soft_weight))
        return weighted_patch, sparse_loss, soft_weight, series_gate


class TemporalCausalityEncoder(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        pred_len,
        series_dim,
        patch_len,
        stride,
        d_model,
        d_ff,
        n_heads,
        e_layers,
        dropout,
        factor,
        activation,
        criterion,
        lambda3,
        lambda3_warmup_steps,
        weight_threshold,
        weight_tau_init,
        weight_tau_min,
        weight_tau_decay,
    ):
        super(TemporalCausalityEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        self.criterion = criterion
        self.exog_dim = enc_in - series_dim
        padding = stride

        self.exog_patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        self.x_patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)

        self.encoder_exg = self._build_encoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            output_attention=True,
            factor=factor,
            e_layers=e_layers,
        )
        self.encoder_x = self._build_encoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            output_attention=False,
            factor=factor,
            e_layers=e_layers,
        )

        self.x_projector = CompressAndProject(self.series_dim, self.seq_len, d_model)
        self.exog_projector = CompressAndProject(max(1, enc_in - series_dim), self.seq_len, d_model)

        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.exog_head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)
        self.x_head = FlattenHead(enc_in, self.head_nf, pred_len, head_dropout=dropout)
        if self.exog_dim > 0:
            self.exog_weight = ExogenousWeightModule(
                exog_dim=self.exog_dim,
                series_dim=self.series_dim,
                d_model=d_model,
                lambda3=lambda3,
                warmup_steps=lambda3_warmup_steps,
                weight_threshold=weight_threshold,
                tau_init=weight_tau_init,
                tau_min=weight_tau_min,
                tau_decay=weight_tau_decay,
            )
        else:
            self.exog_weight = None

    def forward(self, x, exog_future, use_exog=True):
        exog_history = x[:, :, self.series_dim:]
        x_history = x[:, :, :self.series_dim]
        _, _, exog_dim = exog_history.shape
        B, _, _ = x_history.shape

        exog_history_means = exog_history.mean(1, keepdim=True).detach()
        x_history_means = x_history.mean(1, keepdim=True).detach()
        exog_history_stdev = torch.sqrt(torch.var(exog_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_history_stdev = torch.sqrt(torch.var(x_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        exog_history = self.sample_norm(exog_history, exog_history_means, exog_history_stdev)
        x_history = self.sample_norm(x_history, x_history_means, x_history_stdev)

        exog_history = exog_history.permute(0, 2, 1)
        x_history = x_history.permute(0, 2, 1)
        patch_exog, exog_vars = self.exog_patch_embedding(exog_history)
        patch_x, x_vars = self.x_patch_embedding(x_history)

        sparse_loss = torch.tensor(0.0, device=x.device)
        if self.exog_weight is not None and exog_dim > 0:
            patch_exog, sparse_loss, soft_weight, series_gate = self.exog_weight(patch_exog, B, exog_vars)
        else:
            soft_weight = None
            series_gate = torch.ones(B, self.series_dim, device=x.device)

        enc_exog_out, _ = self.encoder_exg(patch_exog)

        if use_exog:
            _, causality_attns = self.encoder_exg(patch_x)
        else:
            causality_attns = None

        exog_history = exog_history.permute(0, 2, 1)
        x_history = x_history.permute(0, 2, 1)

        if exog_dim > 0:
            exog_history_projection = self.exog_projector(exog_history)
        else:
            exog_history_projection = torch.zeros_like(self.x_projector(x_history))
        x_history_projection = self.x_projector(x_history)
        attn_alpha = torch.sigmoid(torch.einsum("bd,bd->b", x_history_projection, exog_history_projection)).view(-1, 1, 1, 1)
        attn_alpha = attn_alpha.repeat(self.series_dim, 1, 1, 1)
        attn_alpha = attn_alpha * series_gate.reshape(-1, 1, 1, 1)

        enc_x_out, _ = self.encoder_x(patch_x, exog_attns=causality_attns, attn_alpha=attn_alpha)
        enc_exog_out = torch.reshape(enc_exog_out, (-1, exog_vars, enc_exog_out.shape[-2], enc_exog_out.shape[-1])).permute(
            0, 1, 3, 2
        )
        enc_x_out = torch.reshape(enc_x_out, (-1, x_vars, enc_x_out.shape[-2], enc_x_out.shape[-1])).permute(0, 1, 3, 2)

        exog_out = self.exog_head(enc_exog_out)
        x_out = self.x_head(enc_x_out)
        exog_out = exog_out.permute(0, 2, 1)
        x_out = x_out.permute(0, 2, 1)
        exog_out = self.sample_denorm(exog_out, exog_history_means, exog_history_stdev)
        x_out = self.sample_denorm(x_out, x_history_means, x_history_stdev)

        if use_exog and exog_future is not None and exog_dim > 0:
            temporal_causality_loss = self.criterion(exog_out, exog_future)
        else:
            temporal_causality_loss = torch.tensor(0.0, device=x.device)

        return x_out, exog_out, temporal_causality_loss, sparse_loss

    def _build_encoder(self, d_model, d_ff, n_heads, dropout, activation, output_attention, factor, e_layers):
        return Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def sample_norm(self, x, means, stdev):
        x = x - means
        x /= stdev
        return x

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x

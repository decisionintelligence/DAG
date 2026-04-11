import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.swan_v3.layers.Embed import CompressAndProject
from ts_benchmark.baselines.swan_v3.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.swan_v3.layers.Transformer_EncDec import Encoder, EncoderLayer


class ExogProjector(nn.Module):
    def __init__(self, input_dim, target_dim, d_model, pred_len):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, target_dim)
        self.sequence_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.feature_proj(x)
        x = x.transpose(1, 2)
        x = self.sequence_proj(x)
        x = x.transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, seq_len, dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.global_embedding = CompressAndProject(dim, seq_len, d_model // 4)
        self.projector = nn.Linear(d_model + d_model // 4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, _, D = x.shape
        global_embedding = self.global_embedding(x).unsqueeze(1).expand(-1, D, -1)
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        x = torch.cat([x, global_embedding], dim=2)
        x = self.projector(x)
        return self.dropout(x)


class FrequencyMahalanobisMask(nn.Module):
    def __init__(self, input_size, tau_init, tau_min, tau_decay, use_phase, phase_weight):
        super(FrequencyMahalanobisMask, self).__init__()
        # SWAN v2: 频域相似性支持幅值+相位联合建模
        frequency_size = input_size // 2 + 1
        self.use_phase = use_phase
        self.phase_weight = phase_weight
        feature_size = frequency_size * (2 if use_phase else 1)
        self.A = nn.Parameter(torch.randn(feature_size, feature_size), requires_grad=True)
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.register_buffer("tau", torch.tensor(tau_init, dtype=torch.float32))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def _update_tau(self):
        if self.training:
            self.step += 1
            self.tau = torch.clamp(self.tau * self.tau_decay, min=self.tau_min)

    def calculate_prob_distance(self, x):
        xf = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(xf)
        if self.use_phase:
            phase = torch.angle(xf)
            feat = torch.cat([amp, self.phase_weight * phase], dim=-1)
        else:
            feat = amp
        diff = feat.unsqueeze(2) - feat.unsqueeze(1)
        q = torch.matmul(self.A.transpose(0, 1), self.A)
        dist = torch.einsum("bijd,dk,bijk->bij", diff, q, diff)
        sim = 1.0 / (dist + 1e-6)
        sim = sim / (sim.max(dim=-1, keepdim=True)[0] + 1e-6)
        eye = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        p = sim * (1 - eye) + eye
        return p.clamp(1e-6, 1 - 1e-6)

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        self._update_tau()
        # SWAN创新：Gumbel-Softmax近似Bernoulli采样，支持可微训练与降温离散化
        logits = torch.log(distribution_matrix) - torch.log(1 - distribution_matrix)
        new_matrix = torch.stack([logits, -logits], dim=-1)
        sampled = F.gumbel_softmax(new_matrix, tau=float(self.tau.item()), hard=True, dim=-1)[..., 0]
        eye = torch.eye(sampled.shape[-1], device=sampled.device).unsqueeze(0)
        sampled = torch.maximum(sampled, eye)
        return sampled

    def forward(self, x):
        distribution = self.calculate_prob_distance(x)
        sample = self.bernoulli_gumbel_rsample(distribution)
        # SWAN创新：输出通道稀疏掩码并返回稀疏度统计用于正则
        mask = sample.unsqueeze(1)
        sparse_loss = sample.mean()
        return mask, sparse_loss


class CovCausalityEncoder(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        pred_len,
        series_dim,
        d_model,
        d_ff,
        n_heads,
        e_layers,
        dropout,
        factor,
        activation,
        criterion,
        mask_tau_init,
        mask_tau_min,
        mask_tau_decay,
        mask_residual_beta,
        freq_use_phase,
        phase_weight,
    ):
        super(CovCausalityEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        self.enc_in = enc_in
        self.exog_dim = enc_in - series_dim
        self.criterion = criterion
        self.mask_residual_beta = mask_residual_beta

        self.history_enc_embedding = DataEmbedding(seq_len, self.exog_dim, d_model, dropout)
        self.future_enc_embedding = DataEmbedding(pred_len, self.exog_dim, d_model, dropout)

        self.future_encoder = self._build_encoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            output_attention=False,
            factor=factor,
            e_layers=e_layers,
        )
        self.history_encoder = self._build_encoder(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            output_attention=True,
            factor=factor,
            e_layers=e_layers,
        )

        self.history_exog_projector = CompressAndProject(self.exog_dim, self.seq_len, d_model)
        self.future_exog_projector = CompressAndProject(self.exog_dim, self.pred_len, d_model)
        self.history_projection = ExogProjector(enc_in - series_dim, 1, d_model, seq_len)
        self.future_projection = ExogProjector(enc_in - series_dim, 1, d_model, pred_len)
        self.mask_generator = FrequencyMahalanobisMask(
            input_size=pred_len,
            tau_init=mask_tau_init,
            tau_min=mask_tau_min,
            tau_decay=mask_tau_decay,
            use_phase=freq_use_phase,
            phase_weight=phase_weight,
        )

    def forward(self, x, exog_future, use_exog=True):
        exog_history = x[:, :, self.series_dim:]
        x_history = x[:, :, :self.series_dim]

        exog_history_means = exog_history.mean(1, keepdim=True).detach()
        exog_history_stdev = torch.sqrt(torch.var(exog_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        exog_history = self.sample_norm(exog_history, exog_history_means, exog_history_stdev)
        exog_future = self.sample_norm(exog_future, exog_history_means, exog_history_stdev)

        history_outs = []
        future_outs = []
        sparse_losses = []

        for i in range(self.series_dim):
            x_history_i = x_history[:, :, i].unsqueeze(-1)
            x_history_i_means = x_history_i.mean(1, keepdim=True).detach()
            x_history_i_stdev = torch.sqrt(torch.var(x_history_i, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

            exog_history_i = self.sample_denorm(
                exog_history,
                x_history_i_means.repeat(1, 1, self.exog_dim),
                x_history_i_stdev.repeat(1, 1, self.exog_dim),
            )
            exog_future_i = self.sample_denorm(
                exog_future,
                x_history_i_means.repeat(1, 1, self.exog_dim),
                x_history_i_stdev.repeat(1, 1, self.exog_dim),
            )

            enc_exog_history_i = self.history_enc_embedding(exog_history_i)
            enc_exog_future_i = self.future_enc_embedding(exog_future_i)

            enc_history_out_i, _ = self.history_encoder(enc_exog_history_i, attn_mask=None)
            if use_exog:
                _, causality_attns_i = self.history_encoder(enc_exog_future_i)
            else:
                causality_attns_i = None

            history_exog_projection_i = self.history_exog_projector(exog_history_i)
            future_exog_projection_i = self.future_exog_projector(exog_future_i)
            attn_alpha_i = torch.sigmoid(torch.einsum("bd,bd->b", history_exog_projection_i, future_exog_projection_i)).view(
                -1, 1, 1, 1
            )

            channel_mask_i, sparse_loss_i = self.mask_generator(exog_future_i.permute(0, 2, 1))
            sparse_losses.append(sparse_loss_i)

            enc_future_out_i, _ = self.future_encoder(
                enc_exog_future_i,
                exog_attns=causality_attns_i,
                attn_alpha=attn_alpha_i,
                attn_mask=None,
                # SWAN创新：将软聚类掩码与残差保真参数注入通道注意力计算
                channel_mask=channel_mask_i,
                residual_beta=self.mask_residual_beta,
            )
            enc_history_out_i = enc_history_out_i.permute(0, 2, 1)
            enc_future_out_i = enc_future_out_i.permute(0, 2, 1)

            history_out_i = self.history_projection(enc_history_out_i)
            future_out_i = self.future_projection(enc_future_out_i)
            history_outs.append(history_out_i)
            future_outs.append(future_out_i)

        history_out = torch.cat(history_outs, dim=-1)
        future_out = torch.cat(future_outs, dim=-1)
        cov_causality_loss = self.criterion(history_out, x_history) if use_exog else torch.tensor(0.0, device=x.device)
        sparse_loss = torch.stack(sparse_losses).mean() if len(sparse_losses) > 0 else torch.tensor(0.0, device=x.device)
        return future_out, cov_causality_loss, sparse_loss

    def _build_encoder(self, d_model, d_ff, n_heads, dropout, activation, output_attention, factor, e_layers):
        return Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
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
            norm_layer=torch.nn.LayerNorm(d_model),
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

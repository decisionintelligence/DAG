import torch
from torch import nn

from ts_benchmark.baselines.swan_v3.layers.CC_EncDec import CovCausalityEncoder
from ts_benchmark.baselines.swan_v3.layers.TC_EncDec import TemporalCausalityEncoder


class DynamicSparseScheduler(nn.Module):
    def __init__(self, beta_min, beta_max, warmup_steps, plateau_patience, step_up, step_down):
        super(DynamicSparseScheduler, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.warmup_steps = max(1, warmup_steps)
        self.plateau_patience = max(1, plateau_patience)
        self.step_up = step_up
        self.step_down = step_down
        self.register_buffer("beta", torch.tensor(beta_min, dtype=torch.float32))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("best_metric", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("no_improve_steps", torch.tensor(0, dtype=torch.long))

    def _update_by_metric(self, metric):
        self.step += 1
        if metric < self.best_metric - 1e-6:
            self.best_metric = metric.detach()
            self.no_improve_steps.zero_()
            self.beta = torch.clamp(self.beta - self.step_down, min=self.beta_min, max=self.beta_max)
        else:
            self.no_improve_steps += 1
            if self.no_improve_steps >= self.plateau_patience:
                self.beta = torch.clamp(self.beta + self.step_up, min=self.beta_min, max=self.beta_max)
                self.no_improve_steps.zero_()

    def step_train(self, metric):
        self._update_by_metric(metric)
        warmup = min(1.0, float(self.step.item()) / float(self.warmup_steps))
        return (self.beta_min + warmup * (self.beta - self.beta_min)).detach()

    def step_validation(self, metric):
        # v3: 使用验证指标更新稀疏调度，提升跨数据集泛化稳定性
        self._update_by_metric(metric)

    def current(self):
        warmup = min(1.0, float(self.step.item()) / float(self.warmup_steps))
        return (self.beta_min + warmup * (self.beta - self.beta_min)).detach()


class SWANV3Model(nn.Module):
    def __init__(self, config):
        super(SWANV3Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.use_c = config.use_c
        self.use_t = config.use_t
        self.use_c_exog = config.use_c_exog
        self.use_t_exog = config.use_t_exog
        self.alpha = config.alpha
        self.beta = config.beta
        self.mask_sparsity_lambda = config.mask_sparsity_lambda
        self.target_mask_density = config.target_mask_density
        self.density_lambda = config.density_lambda
        self.series_dim = config.series_dim
        self.infer_use_future = config.infer_use_future
        self.dynamic_sparse = config.dynamic_sparse
        self.adaptive_alpha = config.adaptive_alpha
        assert self.use_c or self.use_t

        # SWAN创新：时间分支引入外生变量权重学习、软阈值稀疏与温度退火
        self.temporal_encoder = TemporalCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion,
            lambda3=config.lambda3,
            lambda3_warmup_steps=config.lambda3_warmup_steps,
            weight_threshold=config.weight_threshold,
            weight_tau_init=config.weight_tau_init,
            weight_tau_min=config.weight_tau_min,
            weight_tau_decay=config.weight_tau_decay,
            weight_gate_sharpness=config.weight_gate_sharpness,
            weight_gate_floor=config.weight_gate_floor,
        )

        # SWAN创新：通道分支引入频域马氏距离软聚类掩码与残差保真
        self.cov_encoder = CovCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion,
            mask_tau_init=config.mask_tau_init,
            mask_tau_min=config.mask_tau_min,
            mask_tau_decay=config.mask_tau_decay,
            mask_residual_beta=config.mask_residual_beta,
            freq_use_phase=config.freq_use_phase,
            phase_weight=config.phase_weight,
        )
        if self.adaptive_alpha and self.use_c and self.use_t:
            self.alpha_mlp = nn.Sequential(
                nn.Linear(2, config.alpha_hidden),
                nn.GELU(),
                nn.Linear(config.alpha_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.alpha_mlp = None
        if self.dynamic_sparse:
            self.dynamic_scheduler = DynamicSparseScheduler(
                beta_min=config.sparse_beta_min,
                beta_max=config.sparse_beta_max,
                warmup_steps=config.sparse_warmup_steps,
                plateau_patience=config.sparse_plateau_patience,
                step_up=config.sparse_step_up,
                step_down=config.sparse_step_down,
            )
        else:
            self.dynamic_scheduler = None

    def update_validation_metric(self, metric):
        if self.dynamic_scheduler is not None:
            self.dynamic_scheduler.step_validation(metric.detach())

    def _merge_output(self, t_output, c_output):
        if self.use_t and self.use_c:
            if self.alpha_mlp is not None:
                t_stat = t_output.abs().mean(dim=(1, 2), keepdim=False)
                c_stat = c_output.abs().mean(dim=(1, 2), keepdim=False)
                alpha = self.alpha_mlp(torch.stack([t_stat, c_stat], dim=1)).view(-1, 1, 1)
                return alpha * t_output + (1 - alpha) * c_output
            return self.alpha * t_output + (1 - self.alpha) * c_output
        if self.use_t:
            return t_output
        return c_output

    def forward(self, input, exog_future):
        temporal_causality_loss = 0
        cov_causality_loss = 0
        weight_sparse_loss = 0
        mask_sparse_loss = 0
        t_output = None
        c_output = None
        t_exog_output = exog_future

        if not self.training and not self.infer_use_future:
            if self.use_t:
                t_output, t_exog_output, temporal_causality_loss, weight_sparse_loss = self.temporal_encoder(
                    input, None, self.use_t_exog
                )
            if self.use_c:
                c_output, cov_causality_loss, mask_sparse_loss = self.cov_encoder(
                    input, t_exog_output, self.use_c_exog
                )
            output = self._merge_output(t_output, c_output)
            return output, 0

        if self.use_t:
            t_output, t_exog_output, temporal_causality_loss, weight_sparse_loss = self.temporal_encoder(
                input, exog_future, self.use_t_exog
            )
        if self.use_c:
            c_output, cov_causality_loss, mask_sparse_loss = self.cov_encoder(
                input, exog_future, self.use_c_exog
            )

        output = self._merge_output(t_output, c_output)
        causality_loss = self.beta * (temporal_causality_loss + cov_causality_loss)
        density_loss = torch.abs(mask_sparse_loss - self.target_mask_density)
        sparse_loss = weight_sparse_loss + self.mask_sparsity_lambda * mask_sparse_loss + self.density_lambda * density_loss
        if self.dynamic_scheduler is not None:
            sparse_beta = self.dynamic_scheduler.step_train(causality_loss.detach())
            additional_loss = causality_loss + sparse_beta * sparse_loss
        else:
            additional_loss = causality_loss + sparse_loss
        return output, additional_loss

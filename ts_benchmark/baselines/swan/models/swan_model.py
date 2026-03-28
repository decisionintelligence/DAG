from torch import nn

from ts_benchmark.baselines.swan.layers.CC_EncDec import CovCausalityEncoder
from ts_benchmark.baselines.swan.layers.TC_EncDec import TemporalCausalityEncoder


class SWANModel(nn.Module):
    def __init__(self, config):
        super(SWANModel, self).__init__()
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
        self.series_dim = config.series_dim
        self.infer_use_future = config.infer_use_future
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
        )

    def _merge_output(self, t_output, c_output):
        if self.use_t and self.use_c:
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
        # SWAN创新：总附加损失叠加时间稀疏正则与通道掩码稀疏正则
        additional_loss = causality_loss + weight_sparse_loss + self.mask_sparsity_lambda * mask_sparse_loss
        return output, additional_loss

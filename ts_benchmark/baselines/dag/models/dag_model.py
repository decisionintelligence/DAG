from torch import nn

from ts_benchmark.baselines.dag.layers.CC_EncDec import CovCausalityEncoder
from ts_benchmark.baselines.dag.layers.TC_EncDec import TemporalCausalityEncoder


class DAGModel(nn.Module):
    def __init__(self, config):
        super(DAGModel, self).__init__()
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
        self.series_dim = config.series_dim
        self.infer_use_future = config.infer_use_future
        assert self.use_c or self.use_t, "At least one of use_c or use_t must be True"

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
            criterion=config.criterion
        )

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
            criterion=config.criterion

        )

        # self.history_x_projector = CompressAndProject(self.series_dim, self.seq_len, config.d_model)
        # self.future_exog_projector = CompressAndProject(config.enc_in - self.series_dim, self.pred_len, config.d_model)

    def forward(self, input, exog_future):
        # input: [batch_size, seq_len, n_vars]
        temporal_causality_loss = 0
        cov_causality_loss = 0

        if not self.training and not self.infer_use_future:
            if self.use_t:
                t_output, t_exog_output, temporal_causality_loss = self.temporal_encoder(input, None, self.use_t_exog)
            if self.use_c:
                c_output, cov_causality_loss = self.cov_encoder(input, t_exog_output, self.use_c_exog)

            output = self.alpha * t_output + (1 - self.alpha) * c_output
            return output, 0
        else:
            if self.use_t:
                t_output, _, temporal_causality_loss = self.temporal_encoder(input, exog_future, self.use_t_exog)
            if self.use_c:
                c_output, cov_causality_loss = self.cov_encoder(input, exog_future, self.use_c_exog)

            output = self.alpha * t_output + (1 - self.alpha) * c_output
            causality_loss = self.beta * (temporal_causality_loss + cov_causality_loss)

            return output, causality_loss

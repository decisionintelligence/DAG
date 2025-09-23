
import torch
from ts_benchmark.baselines.crosslinear.models.crosslinear_model import CrossLinear_model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "top_k": 5,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "conv_kernel": [12, 16],
    "anomaly_ratio": 1.0,
    "down_sampling_windows": 2,
    "channel_independence": True,
    "down_sampling_layers": 3,
    "down_sampling_method": "avg",
    "decomp_method": "moving_avg",
    "use_norm": True,
    "parallel_strategy": "DP",
    "task_name": "short_term_forecast",
}


class CrossLinear(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super(CrossLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "CrossLinear"

    def _init_model(self):
        return CrossLinear_model(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # decoder input
        dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float()
        dec_input = (
            torch.cat([target[:, : self.config.label_len, :], dec_input], dim=1)
            .float()
            .to(input.device)
        )
        output = self.model(input, input_mark, dec_input, target_mark)

        return {"output": output}
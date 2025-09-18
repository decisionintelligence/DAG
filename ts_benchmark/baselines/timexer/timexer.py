from ts_benchmark.baselines.timexer.model.timexer_model import timexer_model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 64,
    "lradj": "type3",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "input_dim": 20,
    "output_dim": 1,
    "cross_attention_head": 12,
    "cross_attention_dmodel": 96,
    "cross_attention_dropout": 0.1,
    "cross_attention_factor": 2,

    "use_future": 1,

}


class TimeXer(DeepForecastingModelBase):
    """
    TimeKAN adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(TimeXer, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeXer"

    def _init_model(self):
        return timexer_model(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output = self.model(input, exog_future)

        return {"output": output}

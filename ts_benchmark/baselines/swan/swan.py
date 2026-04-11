import torch.nn as nn

from ts_benchmark.baselines.swan.models.swan_model import SWANModel
from ts_benchmark.baselines.utils import DBLoss
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "factor": 1,
    "patch_len": 16,
    "stride": 8,
    "activation": "gelu",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MAE",
    "dbloss_alpha": 0.2,
    "dbloss_beta": 0.5,
    "patience": 10,
    "alpha": 0.2,
    "beta": 0.1,
    # SWAN创新：时间分支外生变量自适应加权与渐进稀疏参数
    "lambda3": 0.01,  # 外生变量权重L1稀疏正则系数
    "lambda3_warmup_steps": 500,  # 稀疏正则从弱到强的预热步数
    "weight_threshold": 0.1,  # 外生变量权重软阈值，低于阈值的连接会被抑制
    "weight_tau_init": 1.5,  # 权重门控温度初值，越大越平滑
    "weight_tau_min": 0.5,  # 权重门控温度下限，防止退火过度
    "weight_tau_decay": 0.999,  # 每步温度衰减率，控制从软选择到稀疏选择的速度
    # SWAN创新：通道分支频域软聚类掩码与残差保真参数
    "mask_tau_init": 5.0,  # Gumbel-Softmax掩码采样温度初值
    "mask_tau_min": 0.5,  # Gumbel温度下限，保留可学习性与稳定性
    "mask_tau_decay": 0.9995,  # Gumbel温度衰减率，控制掩码离散化进程
    "mask_residual_beta": 0.9,  # 掩码注意力占比，剩余部分走全连接残差注意力
    "mask_sparsity_lambda": 0.0,  # 通道掩码稀疏正则系数
    "use_c_exog": True,
    "use_t_exog": True,
    "use_c": True,
    "use_t": True,
    "infer_use_future": True,
}


class SWAN(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(SWAN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "SWAN"

    def _init_criterion(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.config.loss == "DBLoss":
            criterion = DBLoss(self.config.dbloss_alpha, self.config.dbloss_beta)
        else:
            criterion = nn.HuberLoss(delta=0.5)
        self.config.criterion = criterion
        return criterion

    def _init_model(self):
        return SWANModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output, additional_loss = self.model(input, exog_future)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = additional_loss
        return out_loss

import torch.nn as nn
import torch

from ts_benchmark.baselines.swan_v3.models.swan_model import SWANV3Model
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
    "weight_gate_sharpness": 8.0,  # v2: 单阈值平滑门控的斜率
    "weight_gate_floor": 0.05,  # v3: 外生门控最小保留，避免弱有效变量被完全压制
    # SWAN创新：通道分支频域软聚类掩码与残差保真参数
    "mask_tau_init": 5.0,  # Gumbel-Softmax掩码采样温度初值
    "mask_tau_min": 0.5,  # Gumbel温度下限，保留可学习性与稳定性
    "mask_tau_decay": 0.9995,  # Gumbel温度衰减率，控制掩码离散化进程
    "mask_residual_beta": 0.9,  # 掩码注意力占比，剩余部分走全连接残差注意力
    "mask_sparsity_lambda": 0.0,  # 通道掩码稀疏正则系数
    "freq_use_phase": True,  # v2: 频域相似性同时使用幅值和相位
    "phase_weight": 0.5,  # v3: 相位特征权重（0=只幅值，1=相位等权）
    "target_mask_density": 0.35,  # v3: 期望掩码密度
    "density_lambda": 0.1,  # v3: 密度约束损失系数
    "dynamic_sparse": True,  # v2: 稀疏损失动态调度开关
    "sparse_warmup_steps": 500,  # v2: 动态稀疏调度预热步数
    "sparse_plateau_patience": 100,  # v2: plateau判定步数
    "sparse_beta_min": 0.2,  # v2: 稀疏权重下限
    "sparse_beta_max": 1.0,  # v2: 稀疏权重上限
    "sparse_step_up": 0.05,  # v2: 无改进时稀疏权重增幅
    "sparse_step_down": 0.02,  # v2: 有改进时稀疏权重降幅
    "adaptive_alpha": True,  # v3: 时间/通道分支自适应融合开关
    "alpha_hidden": 16,  # v3: 自适应融合MLP隐藏维度
    "use_c_exog": True,
    "use_t_exog": True,
    "use_c": True,
    "use_t": True,
    "infer_use_future": True,
}


class SWANV3(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(SWANV3, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "SWANV3"

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
        return SWANV3Model(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output, additional_loss = self.model(input, exog_future)
        if (not self.model.training) and hasattr(self.model, "update_validation_metric") and target is not None:
            with torch.no_grad():
                # Validation batches may carry label_len + pred_len in target.
                # Align target with model output shape before computing feedback metric.
                aligned_target = target[:, -output.shape[1] :, : output.shape[2]]
                val_metric = self.config.criterion(output, aligned_target)
                self.model.update_validation_metric(val_metric)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = additional_loss
        return out_loss

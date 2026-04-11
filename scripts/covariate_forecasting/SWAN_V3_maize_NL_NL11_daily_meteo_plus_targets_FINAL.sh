#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 maize_NL_NL11_daily_meteo_plus_targets FINAL started ==="

# 数据文件：dataset/forecasting/maize_NL_NL11_daily_otb_meteo_plus_targets_filled.csv
# OTB 顺序：tmin,tmax,tavg,prec,rad,et0,cwb,ssm,rsm，target_channel=-1 对应 rsm
# FINAL 组合（按当前 mse_norm 最优）：
# - h12: BASE
# - h24: TUNE_R5 h24_E3
# - h48: TUNE_R4 h48_D1
# - h96: TUNE_R3 h96_D4

# h12 (BEST from BASE)
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_meteo_plus_targets_filled.csv" --strategy-args '{"horizon": 12, "target_channel": [-1]}' --model-name "swan_v3.SWANV3" --model-hyper-params '{"adaptive_alpha": true, "alpha": 0.6, "alpha_hidden": 16, "batch_size": 64, "d_ff": 256, "d_model": 64, "density_lambda": 0.04, "dropout": 0.0, "dynamic_sparse": true, "e_layers": 1, "freq_use_phase": true, "horizon": 12, "loss": "MAE", "lr": 0.0003, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 120, "patch_len": 8, "patience": 15, "phase_weight": 0.2, "seq_len": 96, "sparse_beta_max": 1.0, "sparse_beta_min": 0.2, "sparse_plateau_patience": 100, "sparse_step_down": 0.02, "sparse_step_up": 0.05, "sparse_warmup_steps": 500, "stride": 8, "target_mask_density": 0.55, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 1, "weight_gate_floor": 0.03, "weight_gate_sharpness": 8.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL" --save-true-pred true

# h24 (BEST from TUNE_R5: h24_E3)
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_meteo_plus_targets_filled.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "swan_v3.SWANV3" --model-hyper-params '{"adaptive_alpha": true, "alpha": 0.6, "alpha_hidden": 16, "batch_size": 64, "d_ff": 256, "d_model": 64, "density_lambda": 0.045, "dropout": 0.01, "dynamic_sparse": true, "e_layers": 1, "freq_use_phase": true, "horizon": 24, "loss": "MAE", "lr": 0.0003, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 120, "patch_len": 8, "patience": 15, "phase_weight": 0.2, "seq_len": 192, "sparse_beta_max": 1.0, "sparse_beta_min": 0.2, "sparse_plateau_patience": 100, "sparse_step_down": 0.02, "sparse_step_up": 0.05, "sparse_warmup_steps": 500, "stride": 8, "target_mask_density": 0.58, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 0, "weight_gate_floor": 0.03, "weight_gate_sharpness": 8.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL" --save-true-pred true

# h48 (BEST from TUNE_R4: h48_D1)
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_meteo_plus_targets_filled.csv" --strategy-args '{"horizon": 48, "target_channel": [-1]}' --model-name "swan_v3.SWANV3" --model-hyper-params '{"adaptive_alpha": true, "alpha": 0.6, "alpha_hidden": 16, "batch_size": 64, "d_ff": 256, "d_model": 64, "density_lambda": 0.055, "dropout": 0.02, "dynamic_sparse": true, "e_layers": 1, "freq_use_phase": true, "horizon": 48, "loss": "MAE", "lr": 0.00025, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 140, "patch_len": 8, "patience": 20, "phase_weight": 0.2, "seq_len": 384, "sparse_beta_max": 1.0, "sparse_beta_min": 0.2, "sparse_plateau_patience": 120, "sparse_step_down": 0.02, "sparse_step_up": 0.05, "sparse_warmup_steps": 600, "stride": 8, "target_mask_density": 0.5, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 0, "weight_gate_floor": 0.03, "weight_gate_sharpness": 8.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL" --save-true-pred true

# h96 (BEST from TUNE_R3: h96_D4)
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_meteo_plus_targets_filled.csv" --strategy-args '{"horizon": 96, "target_channel": [-1]}' --model-name "swan_v3.SWANV3" --model-hyper-params '{"adaptive_alpha": true, "alpha": 0.6, "alpha_hidden": 16, "batch_size": 64, "d_ff": 256, "d_model": 64, "density_lambda": 0.05, "dropout": 0.05, "dynamic_sparse": true, "e_layers": 1, "freq_use_phase": true, "horizon": 96, "loss": "MAE", "lr": 0.0002, "lradj": "type3", "n_heads": 4, "norm": true, "num_epochs": 180, "patch_len": 8, "patience": 30, "phase_weight": 0.2, "seq_len": 768, "sparse_beta_max": 1.0, "sparse_beta_min": 0.2, "sparse_plateau_patience": 140, "sparse_step_down": 0.02, "sparse_step_up": 0.05, "sparse_warmup_steps": 700, "stride": 8, "target_mask_density": 0.45, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 0, "weight_gate_floor": 0.03, "weight_gate_sharpness": 8.0}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL" --save-true-pred true

python ./scripts/plot_forecast_from_logs.py --result-dir "./result/maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL" --out-dir "./result/maize_NL_NL11_daily_otb_meteo_plus_targets/SWAN_V3_FINAL/plots"

echo "=== SWAN_V3 maize_NL_NL11_daily_otb_meteo_plus_targets FINAL finished ==="

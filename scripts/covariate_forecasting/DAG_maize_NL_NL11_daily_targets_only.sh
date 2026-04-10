#!/usr/bin/env bash
set -e

echo "=== DAG maize_NL_NL11_daily_targets_only started ==="

# 数据文件：dataset/forecasting/maize_NL_NL11_daily_otb_targets_only.csv
# OTB 顺序为 ssm -> rsm，因此 target_channel=-1 对应 rsm
# 日频短中期预测：12/24/48

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_targets_only.csv" --strategy-args '{"horizon": 12, "target_channel": [-1]}' --model-name "dag.DAG" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 12, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 96, "stride": 48, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_targets_only/DAG"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_targets_only.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "dag.DAG" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 24, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 192, "stride": 48, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_targets_only/DAG"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "maize_NL_NL11_daily_otb_targets_only.csv" --strategy-args '{"horizon": 48, "target_channel": [-1]}' --model-name "dag.DAG" --model-hyper-params '{"alpha": 0.6, "batch_size": 64, "d_ff": 256, "d_model": 64, "dropout": 0.0, "e_layers": 1, "horizon": 48, "loss": "MAE", "lr": 0.001, "lradj": "type1", "n_heads": 4, "norm": true, "num_epochs": 50, "patch_len": 48, "patience": 5, "seq_len": 384, "stride": 48, "use_c": 1, "use_c_exog": 1, "use_t": 1, "use_t_exog": 1}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "maize_NL_NL11_daily_otb_targets_only/DAG"

echo "=== DAG maize_NL_NL11_daily_targets_only finished ==="

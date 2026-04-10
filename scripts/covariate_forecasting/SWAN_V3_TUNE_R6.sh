#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE R6 started ==="

# R6目标：仅攻克 DE-24
# 思路：围绕 R5 最优 DE_h24_C 做窄范围微调，优先提升稳定性与收敛质量
# 结构固定：seq_len=168, patch_len=48, stride=48, d_model=128, d_ff=64

run_case() {
  local dataset="$1"
  local horizon="$2"
  local seq_len="$3"
  local save_path="$4"
  local alpha="$5"
  local batch_size="$6"
  local d_ff="$7"
  local d_model="$8"
  local dropout="$9"
  local e_layers="${10}"
  local lr="${11}"
  local lradj="${12}"
  local n_heads="${13}"
  local patch_len="${14}"
  local stride="${15:-}"

  local extra_stride=""
  if [ -n "$stride" ]; then
    extra_stride=", \"stride\": $stride"
  fi

  python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "$dataset" \
    --strategy-args "{\"horizon\": $horizon, \"target_channel\": [-1]}" \
    --model-name "swan_v3.SWANV3" \
    --model-hyper-params "{\"alpha\": $alpha, \"batch_size\": $batch_size, \"d_ff\": $d_ff, \"d_model\": $d_model, \"dropout\": $dropout, \"e_layers\": $e_layers, \"horizon\": $horizon, \"loss\": \"MAE\", \"lr\": $lr, \"lradj\": \"$lradj\", \"n_heads\": $n_heads, \"norm\": true, \"num_epochs\": $GROUP_EPOCHS, \"patch_len\": $patch_len, \"patience\": $GROUP_PATIENCE, \"seq_len\": $seq_len$extra_stride, \"use_c\": 1, \"use_c_exog\": 1, \"use_t\": 1, \"use_t_exog\": 1, \"adaptive_alpha\": true, \"alpha_hidden\": 16, \"weight_gate_floor\": $GROUP_GATE_FLOOR, \"weight_gate_sharpness\": 8.0, \"freq_use_phase\": true, \"phase_weight\": $GROUP_PHASE_WEIGHT, \"target_mask_density\": $GROUP_TARGET_DENSITY, \"density_lambda\": $GROUP_DENSITY_LAMBDA, \"dynamic_sparse\": $GROUP_DYNAMIC_SPARSE, \"sparse_warmup_steps\": $GROUP_WARMUP, \"sparse_plateau_patience\": 100, \"sparse_beta_min\": 0.2, \"sparse_beta_max\": 1.0, \"sparse_step_up\": 0.05, \"sparse_step_down\": 0.02}" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "$save_path"
}

# ======================================================
# R6: DE-24 only (9组)
# baseline参考: R5 best = DE_h24_C (mse_norm=0.2882486)
# ======================================================

# ---- Group A: 低学习率 + 长训练 + dynamic_sparse=true ----
GROUP_EPOCHS=220
GROUP_PATIENCE=28
GROUP_PHASE_WEIGHT=0.26
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=900
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_A" 0.7 64 64 128 0.0 1 0.00020 "type3" 4 48 48

GROUP_EPOCHS=240
GROUP_PATIENCE=30
GROUP_PHASE_WEIGHT=0.24
GROUP_TARGET_DENSITY=0.42
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=1000
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_B" 0.7 64 64 128 0.0 1 0.00018 "type3" 4 48 48

GROUP_EPOCHS=200
GROUP_PATIENCE=24
GROUP_PHASE_WEIGHT=0.28
GROUP_TARGET_DENSITY=0.38
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.05
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=1100
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_C" 0.7 64 64 128 0.0 1 0.00022 "type3" 4 48 48

# ---- Group B: dynamic_sparse=false 对照组 ----
GROUP_EPOCHS=220
GROUP_PATIENCE=28
GROUP_PHASE_WEIGHT=0.22
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=700
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_D" 0.7 64 64 128 0.0 1 0.00025 "type3" 4 48 48

GROUP_EPOCHS=240
GROUP_PATIENCE=32
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.48
GROUP_DENSITY_LAMBDA=0.02
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=700
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_E" 0.7 64 64 128 0.0 1 0.00020 "type1" 4 48 48

GROUP_EPOCHS=200
GROUP_PATIENCE=24
GROUP_PHASE_WEIGHT=0.24
GROUP_TARGET_DENSITY=0.43
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=700
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_F" 0.7 64 64 128 0.0 1 0.00030 "type3" 4 48 48

# ---- Group C: 微调 alpha 与 batch_size 的鲁棒性组 ----
GROUP_EPOCHS=220
GROUP_PATIENCE=28
GROUP_PHASE_WEIGHT=0.26
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=900
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_G" 0.75 64 64 128 0.0 1 0.00020 "type3" 4 48 48

GROUP_EPOCHS=220
GROUP_PATIENCE=28
GROUP_PHASE_WEIGHT=0.26
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=900
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_H" 0.65 64 64 128 0.0 1 0.00020 "type3" 4 48 48

GROUP_EPOCHS=220
GROUP_PATIENCE=28
GROUP_PHASE_WEIGHT=0.24
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=900
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R6/DE_h24_I" 0.7 32 64 128 0.0 1 0.00020 "type3" 4 48 48

echo "=== SWAN_V3 TUNE R6 finished ==="

#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE R3 part2 started ==="

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
  local stride="${15}"
  local lambda3="${16:-}"
  local weight_threshold="${17:-}"
  local mask_sparsity_lambda="${18:-}"

  local extra_sparse=""
  if [ -n "$lambda3" ]; then
    extra_sparse=", \"lambda3\": $lambda3, \"weight_threshold\": $weight_threshold, \"mask_sparsity_lambda\": $mask_sparsity_lambda"
  fi

  python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "$dataset" \
    --strategy-args "{\"horizon\": $horizon, \"target_channel\": [-1]}" \
    --model-name "swan_v3.SWANV3" \
    --model-hyper-params "{\"alpha\": $alpha, \"batch_size\": $batch_size, \"d_ff\": $d_ff, \"d_model\": $d_model, \"dropout\": $dropout, \"e_layers\": $e_layers, \"horizon\": $horizon, \"loss\": \"MAE\", \"lr\": $lr, \"lradj\": \"$lradj\", \"n_heads\": $n_heads, \"norm\": true, \"num_epochs\": $GROUP_EPOCHS, \"patch_len\": $patch_len, \"patience\": $GROUP_PATIENCE, \"seq_len\": $seq_len, \"stride\": $stride, \"use_c\": 1, \"use_c_exog\": 1, \"use_t\": 1, \"use_t_exog\": 1, \"adaptive_alpha\": true, \"alpha_hidden\": 16, \"weight_gate_floor\": $GROUP_GATE_FLOOR, \"weight_gate_sharpness\": 8.0, \"freq_use_phase\": true, \"phase_weight\": $GROUP_PHASE_WEIGHT, \"target_mask_density\": $GROUP_TARGET_DENSITY, \"density_lambda\": $GROUP_DENSITY_LAMBDA, \"dynamic_sparse\": $GROUP_DYNAMIC_SPARSE, \"sparse_warmup_steps\": $GROUP_WARMUP, \"sparse_plateau_patience\": 100, \"sparse_beta_min\": 0.2, \"sparse_beta_max\": 1.0, \"sparse_step_up\": 0.05, \"sparse_step_down\": 0.02$extra_sparse}" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "$save_path"
}

# ======================================================
# R3-4: Electricity-192（R2缺失，先补齐并扩展）
# ======================================================
# A: 补跑原 R2-A 思路
GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R3/Electricity_h192_A" 0.5 16 512 256 0.0 1 0.00005 "type3" 4 24 24

# B: 降低学习率并延长训练
GROUP_EPOCHS=150
GROUP_PATIENCE=20
GROUP_PHASE_WEIGHT=0.18
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R3/Electricity_h192_B" 0.5 16 512 256 0.0 1 0.00003 "type3" 4 24 24

# C: 细粒度 patch 对照
GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R3/Electricity_h192_C" 0.5 16 256 128 0.0 1 0.00005 "type3" 4 12 12

echo "=== SWAN_V3 TUNE R3 part2 finished ==="

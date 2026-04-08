#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE R4 part1 started ==="

# R4原则：
# - 仅针对当前仍落后DAG的dataset-horizon点做精调
# - 固定结构参数：seq_len / patch_len / stride / d_model / d_ff
# - 仅调整训练与稀疏相关参数（lr/epochs/patience/phase_weight/density等）

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
# R4-1: BE-24
# 结构固定: seq_len=168, patch_len=8, stride=8, d_model=64, d_ff=256
# ======================================================
GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.22
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "BE.csv" 24 168 "BE/SWAN_V3_TUNE_R4/BE_h24_A" 0.7 64 256 64 0.0 1 0.00005 "type3" 4 8 8

GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.18
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=400
run_case "BE.csv" 24 168 "BE/SWAN_V3_TUNE_R4/BE_h24_B" 0.7 64 256 64 0.0 1 0.0001 "type1" 4 8 8

GROUP_EPOCHS=140
GROUP_PATIENCE=16
GROUP_PHASE_WEIGHT=0.28
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=700
run_case "BE.csv" 24 168 "BE/SWAN_V3_TUNE_R4/BE_h24_C" 0.7 64 256 64 0.0 1 0.00003 "type3" 4 8 8

# ======================================================
# R4-2: BE-360
# 结构固定: seq_len=720, patch_len=8, stride=8, d_model=64, d_ff=256
# ======================================================
GROUP_EPOCHS=140
GROUP_PATIENCE=16
GROUP_PHASE_WEIGHT=0.22
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=600
run_case "BE.csv" 360 720 "BE/SWAN_V3_TUNE_R4/BE_h360_A" 0.7 64 256 64 0.0 1 0.00005 "type3" 4 8 8

GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "BE.csv" 360 720 "BE/SWAN_V3_TUNE_R4/BE_h360_B" 0.7 64 256 64 0.0 1 0.0001 "type1" 4 8 8

GROUP_EPOCHS=160
GROUP_PATIENCE=20
GROUP_PHASE_WEIGHT=0.28
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=800
run_case "BE.csv" 360 720 "BE/SWAN_V3_TUNE_R4/BE_h360_C" 0.7 64 256 64 0.0 1 0.00003 "type3" 4 8 8

# ======================================================
# R4-3: Colbun-30
# 结构固定: seq_len=180, patch_len=30, stride=30, d_model=64, d_ff=64
# ======================================================
GROUP_EPOCHS=150
GROUP_PATIENCE=20
GROUP_PHASE_WEIGHT=0.06
GROUP_TARGET_DENSITY=0.75
GROUP_DENSITY_LAMBDA=0.008
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=200
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R4/Colbun_h30_A" 0.9 64 64 64 0.0 1 0.0003 "type3" 4 30 30

GROUP_EPOCHS=140
GROUP_PATIENCE=18
GROUP_PHASE_WEIGHT=0.08
GROUP_TARGET_DENSITY=0.70
GROUP_DENSITY_LAMBDA=0.01
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=200
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R4/Colbun_h30_B" 0.9 64 64 64 0.0 1 0.0002 "type3" 4 30 30

GROUP_EPOCHS=120
GROUP_PATIENCE=15
GROUP_PHASE_WEIGHT=0.10
GROUP_TARGET_DENSITY=0.65
GROUP_DENSITY_LAMBDA=0.015
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=250
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R4/Colbun_h30_C" 0.9 64 64 64 0.0 1 0.0003 "type1" 4 30 30

# ======================================================
# R4-4: DE-24
# 结构固定: seq_len=168, patch_len=48, stride=48, d_model=128, d_ff=64
# ======================================================
GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.22
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=600
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R4/DE_h24_A" 0.7 64 64 128 0.0 1 0.0005 "type3" 4 48 48

GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.18
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R4/DE_h24_B" 0.7 64 64 128 0.0 1 0.0007 "type1" 4 48 48

GROUP_EPOCHS=140
GROUP_PATIENCE=16
GROUP_PHASE_WEIGHT=0.28
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=700
run_case "DE.csv" 24 168 "DE/SWAN_V3_TUNE_R4/DE_h24_C" 0.7 64 64 128 0.0 1 0.0003 "type3" 4 48 48

# ======================================================
# R4-5: ETTh2-720
# 结构固定: seq_len=96, patch_len=24, stride=24, d_model=256, d_ff=512
# ======================================================
GROUP_EPOCHS=140
GROUP_PATIENCE=18
GROUP_PHASE_WEIGHT=0.18
GROUP_TARGET_DENSITY=0.60
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "ETTh2.csv" 720 96 "ETTh2/SWAN_V3_TUNE_R4/ETTh2_h720_A" 1 64 512 256 0.2 1 0.00005 "type3" 4 24 24

GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.16
GROUP_TARGET_DENSITY=0.65
GROUP_DENSITY_LAMBDA=0.02
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "ETTh2.csv" 720 96 "ETTh2/SWAN_V3_TUNE_R4/ETTh2_h720_B" 1 64 512 256 0.2 1 0.0001 "type1" 4 24 24

GROUP_EPOCHS=150
GROUP_PATIENCE=20
GROUP_PHASE_WEIGHT=0.24
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=700
run_case "ETTh2.csv" 720 96 "ETTh2/SWAN_V3_TUNE_R4/ETTh2_h720_C" 1 64 512 256 0.2 1 0.00005 "type3" 4 24 24

# ======================================================
# R4-6: ETTm2-96
# 结构固定: seq_len=96, patch_len=24, stride=默认(不显式设置), d_model=256, d_ff=512
# ======================================================
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.24
GROUP_TARGET_DENSITY=0.40
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=600
run_case "ETTm2.csv" 96 96 "ETTm2/SWAN_V3_TUNE_R4/ETTm2_h96_A" 1 64 512 256 0.2 2 0.0005 "type3" 4 24

GROUP_EPOCHS=80
GROUP_PATIENCE=8
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "ETTm2.csv" 96 96 "ETTm2/SWAN_V3_TUNE_R4/ETTm2_h96_B" 1 64 512 256 0.2 2 0.0007 "type1" 4 24

GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.28
GROUP_TARGET_DENSITY=0.35
GROUP_DENSITY_LAMBDA=0.08
GROUP_GATE_FLOOR=0.05
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=700
run_case "ETTm2.csv" 96 96 "ETTm2/SWAN_V3_TUNE_R4/ETTm2_h96_C" 1 64 512 256 0.2 2 0.0003 "type3" 4 24

# ======================================================
# R4-7: Electricity-192
# 结构固定: seq_len=96, patch_len=24, stride=24, d_model=256, d_ff=512
# ======================================================
GROUP_EPOCHS=180
GROUP_PATIENCE=24
GROUP_PHASE_WEIGHT=0.16
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.02
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R4/Electricity_h192_A" 0.5 16 512 256 0.0 1 0.00002 "type3" 4 24 24

GROUP_EPOCHS=160
GROUP_PATIENCE=20
GROUP_PHASE_WEIGHT=0.18
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R4/Electricity_h192_B" 0.5 16 512 256 0.0 1 0.00003 "type1" 4 24 24



echo "=== SWAN_V3 TUNE R4 part1 finished ==="

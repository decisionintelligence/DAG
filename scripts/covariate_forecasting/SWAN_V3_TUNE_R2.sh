#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE R2 started ==="

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
# R2-1: Colbun-30（短板点）
# ======================================================
# A: 先去稀疏干扰
GROUP_EPOCHS=80
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.10
GROUP_TARGET_DENSITY=0.65
GROUP_DENSITY_LAMBDA=0.02
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=200
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R2/Colbun_h30_A" 0.9 64 64 128 0.0 1 0.001 "type3" 4 30 30

# B: 提高局部建模分辨率
GROUP_EPOCHS=80
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=200
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R2/Colbun_h30_B" 0.9 64 64 128 0.0 1 0.0005 "type3" 4 15 15

# C: 稳收敛
GROUP_EPOCHS=120
GROUP_PATIENCE=15
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=200
run_case "Colbun.csv" 30 180 "Colbun/SWAN_V3_TUNE_R2/Colbun_h30_C" 0.7 64 64 128 0.0 1 0.001 "type3" 4 30 30

# ======================================================
# R2-2: NP-360（短板点）
# ======================================================
# A: 放松额外稀疏项
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "NP.csv" 360 720 "NP/SWAN_V3_TUNE_R2/NP_h360_A" 0.9 64 128 64 0.0 1 0.001 "type3" 4 48 48 0.0003 0.04 0.0001

# B: 关闭动态稀疏做对照
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.60
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "NP.csv" 360 720 "NP/SWAN_V3_TUNE_R2/NP_h360_B" 0.9 64 128 64 0.0 1 0.001 "type3" 4 48 48

# C: 增强长依赖
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "NP.csv" 360 960 "NP/SWAN_V3_TUNE_R2/NP_h360_C" 0.9 64 256 128 0.0 1 0.001 "type3" 4 32 32

# ======================================================
# R2-3: Energy-360（短板点）
# ======================================================
# A: 学习率/调度修正
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Energy.csv" 360 720 "Energy/SWAN_V3_TUNE_R2/Energy_h360_A" 0.6 64 256 64 0.0 1 0.0003 "type3" 4 48 48

# B: 容量略增
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Energy.csv" 360 720 "Energy/SWAN_V3_TUNE_R2/Energy_h360_B" 0.6 64 512 128 0.1 1 0.001 "type1" 8 48 48

# C: 减弱结构正则
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.60
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Energy.csv" 360 720 "Energy/SWAN_V3_TUNE_R2/Energy_h360_C" 0.6 64 256 64 0.0 1 0.001 "type1" 4 48 48

# ======================================================
# R2-4: Weather-96（短板点）
# ======================================================
# A: 短期更细粒度 patch
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Weather.csv" 96 96 "Weather/SWAN_V3_TUNE_R2/Weather_h96_A" 0.5 32 256 128 0.0 1 0.0001 "type3" 4 12 12

# B: 减少稀疏约束
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.10
GROUP_TARGET_DENSITY=0.60
GROUP_DENSITY_LAMBDA=0.02
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Weather.csv" 96 96 "Weather/SWAN_V3_TUNE_R2/Weather_h96_B" 0.5 32 512 256 0.0 1 0.0001 "type3" 4 24 24

# C: 防过拟合
GROUP_EPOCHS=80
GROUP_PATIENCE=8
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Weather.csv" 96 96 "Weather/SWAN_V3_TUNE_R2/Weather_h96_C" 0.5 32 512 256 0.1 1 0.0002 "type3" 4 24 24

# ======================================================
# R2-5: Weather-336（短板点）
# ======================================================
# A: 增加上下文
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Weather.csv" 336 192 "Weather/SWAN_V3_TUNE_R2/Weather_h336_A" 0.5 32 512 256 0.0 1 0.0001 "type3" 4 24 24

# B: 中长期增强周期项
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.30
GROUP_TARGET_DENSITY=0.50
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Weather.csv" 336 96 "Weather/SWAN_V3_TUNE_R2/Weather_h336_B" 0.5 32 512 256 0.0 1 0.0001 "type3" 4 24 24

# C: 更粗 patch 降噪
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Weather.csv" 336 96 "Weather/SWAN_V3_TUNE_R2/Weather_h336_C" 0.5 32 512 256 0.1 1 0.0001 "type3" 4 48 24

# ======================================================
# R2-6: Electricity-192（短板点）
# ======================================================
# A: 稳学习率
GROUP_EPOCHS=120
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R2/Electricity_h192_A" 0.5 16 512 256 0.0 1 0.00005 "type3" 4 24 24

# B: 短中期细粒度
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.20
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R2/Electricity_h192_B" 0.5 16 256 128 0.0 1 0.0001 "type3" 4 12 12

# C: 放松稀疏
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.03
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500
run_case "Electricity.csv" 192 96 "Electricity/SWAN_V3_TUNE_R2/Electricity_h192_C" 0.5 16 512 256 0.0 1 0.0001 "type3" 4 24 24

echo "=== SWAN_V3 TUNE R2 finished ==="

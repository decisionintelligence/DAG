#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE part2 started ==="

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

# 推荐先跑的优先级顺序：第4B组（ETTh2/ETTm2）→ 第1组（负荷/能源）→ 第2组（风电）→ 第5组（大规模多变量）→ 第3组（水电短预测）→ 第4A组（ETTh1/ETTm1）。

# 第4A组：ETTh1 / ETTm1（表现较好组，做“稳态增强”）
# 作用：在已有较好结果的配置基础上继续挖掘上限，作为高性能参考组。
# 目的：
# 1) 保留较强模型容量与稀疏策略（dynamic_sparse=true），防止破坏已验证有效的结构增益。
# 2) 增加训练轮次与 patience，提升长 horizon（336/720）下的稳定性。
# 3) phase_weight 提高到 0.45（低于原 ETTh1 的激进值），在性能与泛化之间取折中。
GROUP_EPOCHS=100
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.45
GROUP_TARGET_DENSITY=0.4
GROUP_DENSITY_LAMBDA=0.06
GROUP_GATE_FLOOR=0.04
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500

for horizon in 96 192 336 720; do
  seq_len=96
  run_case "ETTh1.csv" "$horizon" "$seq_len" "ETTh1/SWAN_V3_TUNE" 0.95 64 512 1024 0.1 2 0.0001 "type3" 2 24 24
  run_case "ETTm1.csv" "$horizon" "$seq_len" "ETTm1/SWAN_V3_TUNE" 1 64 512 256 0.3 3 1e-05 "type3" 16 24 24
done

# 第5组：大规模多变量数据（Weather、Electricity、Traffic、Exchange）
# 作用：处理通道数多、相关性复杂的数据集，兼顾训练可行性与泛化能力。
# 目的：
# 1) 采用中等稀疏密度（0.45）和中等稀疏惩罚（0.05），避免高维场景下过强剪枝。
# 2) phase_weight 设为 0.2，减少在多变量耦合场景中对相位特征的过度依赖。
# 3) 保留 dynamic_sparse=true 与较长 warmup，兼顾训练早期稳定和后期压缩能力。
# 4) 延续各数据集原始 batch_size 差异，控制显存压力并保持与历史结果可比性。
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.2
GROUP_TARGET_DENSITY=0.45
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500

for horizon in 96 192 336 720; do
  seq_len=96
  run_case "Weather.csv" "$horizon" "$seq_len" "Weather/SWAN_V3_TUNE" 0.5 32 512 256 0.0 1 0.0001 "type3" 4 24 24
  run_case "Electricity.csv" "$horizon" "$seq_len" "Electricity/SWAN_V3_TUNE" 0.5 16 512 256 0.0 1 0.0001 "type3" 4 24 24
  run_case "Traffic.csv" "$horizon" "$seq_len" "Traffic/SWAN_V3_TUNE" 0.2 32 512 256 0.0 1 0.0001 "type3" 4 24 24
  run_case "Exchange.csv" "$horizon" "$seq_len" "Exchange/SWAN_V3_TUNE" 0.9 64 512 256 0.3 2 0.0001 "type3" 16 24 24
done

echo "=== SWAN_V3 TUNE part2 finished ==="

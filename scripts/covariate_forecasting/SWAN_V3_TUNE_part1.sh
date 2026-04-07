#!/usr/bin/env bash
set -e

echo "=== SWAN_V3 TUNE part1 started ==="

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

# 第1组：负荷/能源类数据（NP、PJM、BE、FR、DE、Energy）
# 作用：作为多数业务场景的主力配置组，重点处理“中等周期 + 长期预测并存”的负荷波动。
# 目的：
# 1) 将训练轮次提升到 100、早停耐心提高到 10，避免稀疏门控尚未稳定就提前停止。
# 2) 将 target_mask_density 提高到 0.5、density_lambda 降到 0.05，缓解过度稀疏导致的有效信息丢失。
# 3) 将 phase_weight 控制在 0.2，减少相位项在非强周期片段上的过拟合风险。
# 4) NP 单独保留 lambda3/threshold/sparsity_lambda 的轻量约束，用于稳定外生权重选择。
GROUP_EPOCHS=100
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.2
GROUP_TARGET_DENSITY=0.5
GROUP_DENSITY_LAMBDA=0.05
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=500

for horizon in 24 360; do
  if [ "$horizon" -eq 24 ]; then seq_len=168; else seq_len=720; fi

  if [ "$horizon" -eq 24 ]; then
    run_case "NP.csv" "$horizon" "$seq_len" "NP/SWAN_V3_TUNE" 0.9 64 128 64 0.0 1 0.001 "type3" 4 48 48 0.0005 0.05 0.0001
  else
    run_case "NP.csv" "$horizon" "$seq_len" "NP/SWAN_V3_TUNE" 0.9 64 128 64 0.0 1 0.001 "type3" 4 48 48 0.001 0.06 0.0003
  fi

  run_case "PJM.csv" "$horizon" "$seq_len" "PJM/SWAN_V3_TUNE" 0.7 64 256 64 0.0 1 0.001 "type1" 4 48 48
  run_case "BE.csv" "$horizon" "$seq_len" "BE/SWAN_V3_TUNE" 0.7 64 256 64 0.0 1 0.0001 "type3" 4 8 8
  run_case "FR.csv" "$horizon" "$seq_len" "FR/SWAN_V3_TUNE" 0.5 64 256 64 0.0 1 0.0001 "type3" 4 48 48
  run_case "DE.csv" "$horizon" "$seq_len" "DE/SWAN_V3_TUNE" 0.7 64 64 128 0.0 1 0.001 "type3" 4 48 48
  run_case "Energy.csv" "$horizon" "$seq_len" "Energy/SWAN_V3_TUNE" 0.6 64 256 64 0.0 1 0.001 "type1" 4 48 48
done

# 第2组：风电数据（Sdwpfm1、Sdwpfm2、Sdwpfh1、Sdwpfh2）
# 作用：面向高噪声、随机扰动更强的风电序列，提升模型在突变工况下的鲁棒性。
# 目的：
# 1) phase_weight 下调到 0.15，避免频域相位项放大噪声。
# 2) target_mask_density 设为 0.5 且 density_lambda=0.04，先保留更多通道信息，再由训练自行收缩。
# 3) sparse_warmup_steps 缩短到 300，让稀疏策略更早参与，但仍保留足够缓冲避免训练震荡。
# 4) dropout 维持在 0.05（按数据集调用处体现），抑制噪声驱动的过拟合。
GROUP_EPOCHS=100
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.5
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=300

for horizon in 24 360; do
  if [ "$horizon" -eq 24 ]; then seq_len=168; else seq_len=720; fi

  run_case "Sdwpfm1.csv" "$horizon" "$seq_len" "Sdwpfm1/SWAN_V3_TUNE" 0.3 64 64 64 0.05 1 0.001 "type3" 4 8 8
  run_case "Sdwpfm2.csv" "$horizon" "$seq_len" "Sdwpfm2/SWAN_V3_TUNE" 0.1 64 512 64 0.05 1 0.001 "type3" 4 48 48
  run_case "Sdwpfh1.csv" "$horizon" "$seq_len" "Sdwpfh1/SWAN_V3_TUNE" 0.05 64 64 64 0.05 1 0.001 "type3" 4 8 8
  run_case "Sdwpfh2.csv" "$horizon" "$seq_len" "Sdwpfh2/SWAN_V3_TUNE" 0.05 64 512 64 0.05 1 0.001 "type3" 4 8 8
done

# 第3组：水电短预测数据（Colbun、Rapel）
# 作用：针对短 horizon（10/30）的快速预测任务，强调收敛速度与稳定性。
# 目的：
# 1) 训练轮次设置为 80，避免在小样本短窗口任务上无效拉长训练。
# 2) warmup 缩短为 200，使稀疏机制尽快生效，提升短周期任务的收敛效率。
# 3) 维持较温和的相位与稀疏强度（phase=0.15, density_lambda=0.04），减少过正则带来的欠拟合。
GROUP_EPOCHS=80
GROUP_PATIENCE=10
GROUP_PHASE_WEIGHT=0.15
GROUP_TARGET_DENSITY=0.5
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=true
GROUP_WARMUP=200

for horizon in 10 30; do
  if [ "$horizon" -eq 10 ]; then seq_len=60; else seq_len=180; fi

  run_case "Colbun.csv" "$horizon" "$seq_len" "Colbun/SWAN_V3_TUNE" 0.9 64 64 128 0.0 1 0.001 "type3" 4 30 30
  run_case "Rapel.csv" "$horizon" "$seq_len" "Rapel/SWAN_V3_TUNE" 0.6 64 64 128 0.0 1 0.001 "type3" 4 8 8
done

# 第4B组：ETTh2 / ETTm2（表现波动组，先做“保守稀疏”）
# 作用：先验证 SWAN_V3 的净结构收益，再逐步恢复动态稀疏，降低调参不稳定性。
# 目的：
# 1) dynamic_sparse 先关闭（false），排查“动态稀疏策略”是否是性能波动主因。
# 2) target_mask_density 提高到 0.55，优先保证信息保留，避免错误剪枝。
# 3) phase_weight 降至 0.2，降低相位项对复杂季节性/噪声混合序列的干扰。
# 4) 后续若该组改善，可在此基础上再开启 dynamic_sparse 做第二阶段精调。
GROUP_EPOCHS=100
GROUP_PATIENCE=12
GROUP_PHASE_WEIGHT=0.2
GROUP_TARGET_DENSITY=0.55
GROUP_DENSITY_LAMBDA=0.04
GROUP_GATE_FLOOR=0.03
GROUP_DYNAMIC_SPARSE=false
GROUP_WARMUP=500

for horizon in 96 192 336 720; do
  seq_len=96
  run_case "ETTh2.csv" "$horizon" "$seq_len" "ETTh2/SWAN_V3_TUNE" 1 64 512 256 0.2 1 0.0001 "type3" 4 24 24
  run_case "ETTm2.csv" "$horizon" "$seq_len" "ETTm2/SWAN_V3_TUNE" 1 64 512 256 0.2 2 0.001 "type3" 4 24 24
done


echo "=== SWAN_V3 TUNE part1 finished ==="

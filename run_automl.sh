#!/usr/bin/env bash
# =============================================================
# ForeAct AutoML 运行脚本
# =============================================================
# 用法:
#   # pure_orig 模式（纯完整orig数据 + CoT，新的wandb项目和study）
#   bash run_automl.sh pure_orig
#
#   # mixed 模式（精选orig + subtask，沿用旧配置）
#   bash run_automl.sh mixed
#
#   # 默认为 pure_orig
#   bash run_automl.sh
# =============================================================

set -euo pipefail

# ---- 参数 ----
DATA_MODE="${1:-pure_orig}"
N_TRIALS="${2:-50}"
NUM_EPOCHS="${3:-5}"
GPUS="${4:-8}"
VLM_MODEL="${5:-doubao-seed-2-0-lite-260215}"

# ---- 环境变量 ----
export VOLCENKEY="${VOLCENKEY:-8dc53ee3-fddc-47b5-b419-5888d5e057ff}"

# ---- 路径 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 日志文件 ----
if [[ "$DATA_MODE" == "pure_orig" ]]; then
    LOG_FILE="automl_pureorig_run.log"
else
    LOG_FILE="automl_run.log"
fi

echo "============================================================"
echo "ForeAct AutoML 启动"
echo "  数据模式:    $DATA_MODE"
echo "  搜索轮数:    $N_TRIALS"
echo "  Epoch 数:    $NUM_EPOCHS"
echo "  GPU 数量:    $GPUS"
echo "  VLM 模型:    $VLM_MODEL"
echo "  日志文件:    $LOG_FILE"
echo "============================================================"

nohup python3 automl_optuna.py \
    --vlm_model "$VLM_MODEL" \
    --data_mode "$DATA_MODE" \
    --n_trials "$N_TRIALS" \
    --num_epochs "$NUM_EPOCHS" \
    --gpus "$GPUS" \
    > "$LOG_FILE" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志: tail -f $LOG_FILE"

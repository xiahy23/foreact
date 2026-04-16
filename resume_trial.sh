#!/usr/bin/env bash
# =============================================================
# 恢复被中断的 AutoML trial 训练
# 用法:
#   bash resume_trial.sh 001   # 恢复 trial_001 (只剩10步)
#   bash resume_trial.sh 000   # 恢复 trial_000
# =============================================================
set -euo pipefail

TRIAL_ID="${1:?用法: bash resume_trial.sh <trial_id, e.g. 001>}"
GPUS="${2:-8}"
MASTER_PORT="${3:-25020}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="automl_trials/trial_${TRIAL_ID}_resume.yaml"
LOG_FILE="resume_trial_${TRIAL_ID}.log"

if [[ ! -f "configs/${CONFIG_FILE}" ]]; then
    echo "错误: configs/${CONFIG_FILE} 不存在"
    exit 1
fi

export WANDB_PROJECT="VisualForesight"

echo "============================================================"
echo "恢复 Trial ${TRIAL_ID} 训练"
echo "  配置文件:    configs/${CONFIG_FILE}"
echo "  GPU 数量:    ${GPUS}"
echo "  Master Port: ${MASTER_PORT}"
echo "  日志文件:    ${LOG_FILE}"
echo "============================================================"

nohup torchrun \
    --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} \
    train.py \
    --config_file "${CONFIG_FILE}" \
    > "${LOG_FILE}" 2>&1 &

echo "后台进程 PID: $!"
echo "查看日志: tail -f ${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export FOREACT_PYAV_CONTAINER_CACHE_SIZE="${FOREACT_PYAV_CONTAINER_CACHE_SIZE:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_FILE="F-exp41-wipe_cup_towel_from_F-exp37_s1_t2s_bs64x1_e20.yaml"
RUN_NAME="F-exp41-wipe_cup_towel_from_F-exp37_s1_t2s_bs64x1_e20_nofilter"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-26041}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/aloha_fexp
LOG_FILE="${LOG_FILE:-logs/aloha_fexp/${RUN_NAME}_${RUN_TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] log=${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] config=configs/${CONFIG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

conda run --no-capture-output -n foreact \
  accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MAIN_PROCESS_PORT" \
    --mixed_precision bf16 \
    train.py \
    --config_file "$CONFIG_FILE"

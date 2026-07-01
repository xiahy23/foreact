#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export FOREACT_PYAV_CONTAINER_CACHE_SIZE="${FOREACT_PYAV_CONTAINER_CACHE_SIZE:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_FILE="F-exp33-food-nonfood_ep120-139_taillast_from_F-exp32_e20.yaml"
RUN_NAME="F-exp33-food-nonfood_ep120-139_taillast_from_F-exp32_e20"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-26033}"
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

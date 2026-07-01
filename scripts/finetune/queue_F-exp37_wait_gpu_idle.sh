#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

RUN_NAME="F-exp37-aloha_wipe_plate_cup_gripper_binary_motion_delta005_s1_t2s_bs16x8_e5"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/aloha_fexp
LOG_FILE="${LOG_FILE:-logs/aloha_fexp/queue_${RUN_NAME}_${RUN_TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue log=${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting until first ${NUM_GPUS} GPUs each have >= ${MIN_FREE_MB} MiB free"

while true; do
  mapfile -t FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  READY=1
  SUMMARY=""
  for ((i=0; i<NUM_GPUS; i++)); do
    value="${FREE_MB[$i]:-0}"
    SUMMARY+="${i}:${value}MiB "
    if (( value < MIN_FREE_MB )); then
      READY=0
    fi
  done

  if (( READY == 1 )); then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs ready: ${SUMMARY}"
    break
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs not idle yet: ${SUMMARY}; sleeping ${CHECK_INTERVAL}s"
  sleep "${CHECK_INTERVAL}"
done

exec scripts/finetune/F-exp37-aloha_wipe_plate_cup_gripper_binary_motion_delta005_s1_t2s_bs16x8_e5.sh

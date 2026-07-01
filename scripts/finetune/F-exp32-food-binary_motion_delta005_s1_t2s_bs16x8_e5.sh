#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export FOREACT_PYAV_CONTAINER_CACHE_SIZE="${FOREACT_PYAV_CONTAINER_CACHE_SIZE:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

CONFIG_FILE="F-exp32-food-binary_motion_delta005_s1_t2s_bs16x8_e5.yaml"
RUN_NAME="F-exp32-food-binary_motion_delta005_s1_t2s_bs16x8_e5"
NUM_GPUS="${NUM_GPUS:-8}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-26032}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/aloha_fexp
LOG_FILE="${LOG_FILE:-logs/aloha_fexp/${RUN_NAME}_${RUN_TS}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

select_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v min_free="$MIN_FREE_MB" -v max_util="$MAX_GPU_UTIL" -v want="$NUM_GPUS" '
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          gsub(/ /, "", $3);
          if ($2 >= min_free && $3 <= max_util && n < want) {
            n++;
            ids[n] = $1;
          }
        }
        END {
          for (i = 1; i <= n; i++) {
            printf "%s%s", (i == 1 ? "" : ","), ids[i];
          }
          if (n > 0) {
            printf "\n";
          }
        }'
}

csv_count() {
  local csv="$1"
  if [[ -z "$csv" ]]; then
    echo 0
  else
    awk -F, '{print NF}' <<< "$csv"
  fi
}

wait_for_gpus() {
  local selected
  local count
  while true; do
    selected="$(select_gpus)"
    count="$(csv_count "$selected")"
    if (( count >= NUM_GPUS )); then
      echo "$selected"
      return 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for ${NUM_GPUS} GPUs with >=${MIN_FREE_MB}MiB free and <=${MAX_GPU_UTIL}% util; currently selected=${selected:-none}" >&2
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] log=${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] config=configs/${CONFIG_FILE}"

selected_gpus="$(wait_for_gpus)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] selected GPUs: ${selected_gpus}"

CUDA_VISIBLE_DEVICES="$selected_gpus" \
  conda run --no-capture-output -n foreact \
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --main_process_port "$MAIN_PROCESS_PORT" \
      --mixed_precision bf16 \
      train.py \
      --config_file "$CONFIG_FILE"

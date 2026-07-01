#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_GPUS="${NUM_GPUS:-8}"
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE="F-exp31-three-object-lerobot-binary_motion_delta005_s5.yaml"
RUN_LABEL="F-exp31-three-object-lerobot-binary_motion_delta005_s5"
TRAIN_PORT="${TRAIN_PORT:-26034}"
ACT_HOLD_PATH="${ACT_HOLD_PATH:-/media/raid/workspace/xiahongyu/Agent-VLA/playground/Datasets/act_coy.py}"
ACT_COT_FALLBACK="/media/raid/workspace/xiahongyu/Agent-VLA/playground/Datasets/act_cot.py"
HOLD_UTIL="${HOLD_UTIL:-80,80,80,80,80,80,80,80}"

mkdir -p logs/aloha_fexp
QUEUE_LOG="${QUEUE_LOG:-logs/aloha_fexp/queue_F-exp31_three_object_g8_then_act_cot.log}"
exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue log=${QUEUE_LOG}"

if [[ ! -f "$ACT_HOLD_PATH" && -f "$ACT_COT_FALLBACK" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${ACT_HOLD_PATH} not found; using ${ACT_COT_FALLBACK}"
  ACT_HOLD_PATH="$ACT_COT_FALLBACK"
fi

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

selected_gpus="$(wait_for_gpus)"
train_log="logs/aloha_fexp/${RUN_LABEL}_${RUN_TS}.log"
hold_log="logs/aloha_fexp/act_cot_hold_after_${RUN_LABEL}_${RUN_TS}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${RUN_LABEL} on CUDA_VISIBLE_DEVICES=${selected_gpus}; log=${train_log}"

env CUDA_VISIBLE_DEVICES="$selected_gpus" \
  conda run --no-capture-output -n foreact \
    accelerate launch \
      --num_processes "$NUM_GPUS" \
      --main_process_port "$TRAIN_PORT" \
      --mixed_precision bf16 \
      train.py \
      --config_file "$CONFIG_FILE" \
  > "$train_log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished ${RUN_LABEL}; starting GPU hold with ${ACT_HOLD_PATH}; log=${hold_log}"
unset CUDA_VISIBLE_DEVICES
exec conda run --no-capture-output -n foreact \
  python "$ACT_HOLD_PATH" --util "$HOLD_UTIL" \
  > "$hold_log" 2>&1

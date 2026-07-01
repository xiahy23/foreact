#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_GPUS_PER_EXP="${NUM_GPUS_PER_EXP:-4}"
TOTAL_GPUS=$((NUM_GPUS_PER_EXP * 2))
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/aloha_fexp

select_gpus() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v min_free="$MIN_FREE_MB" -v max_util="$MAX_GPU_UTIL" -v want="$TOTAL_GPUS" '
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

csv_slice() {
  local csv="$1"
  local start="$2"
  local len="$3"
  local -a ids
  local -a out=()
  local i

  IFS=',' read -r -a ids <<< "$csv"
  for ((i = start; i < start + len; i++)); do
    out+=("${ids[$i]}")
  done
  local IFS=,
  echo "${out[*]}"
}

wait_for_gpus() {
  local selected
  local count
  while true; do
    selected="$(select_gpus)"
    count="$(csv_count "$selected")"
    if (( count >= TOTAL_GPUS )); then
      echo "$selected"
      return 0
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for ${TOTAL_GPUS} GPUs with >=${MIN_FREE_MB}MiB free and <=${MAX_GPU_UTIL}% util; currently selected=${selected:-none}" >&2
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

run_experiment() {
  local config_file="$1"
  local port="$2"
  local run_label="$3"
  local gpus="$4"
  local log_file="logs/aloha_fexp/${run_label}_parallel_${RUN_TS}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting ${run_label} on CUDA_VISIBLE_DEVICES=${gpus}; log=${log_file}"

  env CUDA_VISIBLE_DEVICES="$gpus" \
    conda run --no-capture-output -n foreact \
      accelerate launch \
        --num_processes "$NUM_GPUS_PER_EXP" \
        --main_process_port "$port" \
        --mixed_precision bf16 \
        train.py \
        --config_file "$config_file" \
    > "$log_file" 2>&1

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished ${run_label}"
}

selected_gpus="$(wait_for_gpus)"
gpus_left="$(csv_slice "$selected_gpus" 0 "$NUM_GPUS_PER_EXP")"
gpus_right="$(csv_slice "$selected_gpus" "$NUM_GPUS_PER_EXP" "$NUM_GPUS_PER_EXP")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] selected GPUs: ${selected_gpus}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] left_first GPUs: ${gpus_left}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] right_first GPUs: ${gpus_right}"

run_experiment \
  "F-exp28-obstacle-left-first_motion_delta005_s5_t2s.yaml" \
  "26030" \
  "F-exp28-obstacle-left-first_motion_delta005_s5_t2s" \
  "$gpus_left" &
pid_left=$!

run_experiment \
  "F-exp29-obstacle-right-first_motion_delta005_s5_t2s.yaml" \
  "26031" \
  "F-exp29-obstacle-right-first_motion_delta005_s5_t2s" \
  "$gpus_right" &
pid_right=$!

status_left=0
status_right=0

wait "$pid_left" || status_left=$?
wait "$pid_right" || status_right=$?

if (( status_left != 0 || status_right != 0 )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] training failed: left_first=${status_left}, right_first=${status_right}" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] all obstacle left/right training jobs finished"

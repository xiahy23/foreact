#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact

export FORCE_VIDEO_BACKEND="${FORCE_VIDEO_BACKEND:-pyav}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NUM_GPUS_PER_EXP="${NUM_GPUS_PER_EXP:-4}"
NUM_EXPERIMENTS=2
TOTAL_GPUS=$((NUM_GPUS_PER_EXP * NUM_EXPERIMENTS))
MIN_FREE_MB="${MIN_FREE_MB:-60000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

ACT_COT_PATH="${ACT_COT_PATH:-/media/raid/workspace/xiahongyu/Agent-VLA/playground/Datasets/act_cot.py}"
HOLD_UTIL="${HOLD_UTIL:-80,80,80,80,80,80,80,80}"

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

start_hold() {
  local log_file="logs/aloha_fexp/act_cot_hold_${RUN_TS}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] both experiments finished; starting act_cot hold on all visible GPUs; log=${log_file}"
  unset CUDA_VISIBLE_DEVICES
  exec conda run --no-capture-output -n foreact \
    python "$ACT_COT_PATH" --util "$HOLD_UTIL" \
    > "$log_file" 2>&1
}

selected_gpus="$(wait_for_gpus)"
gpus_exp23="$(csv_slice "$selected_gpus" 0 "$NUM_GPUS_PER_EXP")"
gpus_exp24="$(csv_slice "$selected_gpus" "$NUM_GPUS_PER_EXP" "$NUM_GPUS_PER_EXP")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] selected GPUs: ${selected_gpus}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] F-exp23 GPUs: ${gpus_exp23}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] F-exp24 GPUs: ${gpus_exp24}"

run_experiment \
  "F-exp23-eggplant-potato-gripper-binary_motion_delta005_s5.yaml" \
  "26023" \
  "F-exp23-eggplant-potato-gripper-binary_motion_delta005_s5" \
  "$gpus_exp23" &
pid_exp23=$!

run_experiment \
  "F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5.yaml" \
  "26024" \
  "F-exp24-aloha-banana-fix-cube-eggplant-potato_motion_delta005_s5" \
  "$gpus_exp24" &
pid_exp24=$!

status_exp23=0
status_exp24=0

wait "$pid_exp23" || status_exp23=$?
wait "$pid_exp24" || status_exp24=$?

if (( status_exp23 != 0 || status_exp24 != 0 )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] training failed: F-exp23=${status_exp23}, F-exp24=${status_exp24}; act_cot hold will not start" >&2
  exit 1
fi

start_hold

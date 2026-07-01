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
CACHE_WORKERS="${CACHE_WORKERS:-24}"
CACHE_ROOT="${CACHE_ROOT:-datasets/frame_cache_cam_high}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

CONFIG_EXP30="F-exp30-EAI-lerobot_motion_delta005_offset60_s1_e2.yaml"
CONFIG_EXP31="F-exp31-three-object-lerobot-binary_motion_delta005_offset60_s1_e2.yaml"
RUN_EXP30="F-exp30-EAI-lerobot_motion_delta005_offset60_s1_e2"
RUN_EXP31="F-exp31-three-object-lerobot-binary_motion_delta005_offset60_s1_e2"

mkdir -p logs/aloha_fexp
QUEUE_LOG="${QUEUE_LOG:-logs/aloha_fexp/queue_F-exp30_F-exp31_offset60_s1_e2_parallel_cached_${RUN_TS}.log}"
exec > >(tee -a "$QUEUE_LOG") 2>&1

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

build_cache() {
  local dataset_root="$1"
  local dataset_name="$2"
  local log_file="logs/aloha_fexp/frame_cache_${dataset_name}_${RUN_TS}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] building frame cache for ${dataset_name}; log=${log_file}"
  conda run --no-capture-output -n foreact \
    python scripts/build_frame_cache.py \
      --dataset-root "$dataset_root" \
      --cache-root "$CACHE_ROOT" \
      --dataset-name "$dataset_name" \
      --camera-key observation.images.cam_high \
      --workers "$CACHE_WORKERS" \
      --quality 92 \
    > "$log_file" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished frame cache for ${dataset_name}"
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue log=${QUEUE_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] cache root=${CACHE_ROOT}, cache workers=${CACHE_WORKERS}"

build_cache "datasets/EAI_lerobot" "EAI_lerobot" &
pid_cache30=$!
build_cache "datasets/three_object_lerobot_binary" "three_object_lerobot_binary" &
pid_cache31=$!

status_cache30=0
status_cache31=0
wait "$pid_cache30" || status_cache30=$?
wait "$pid_cache31" || status_cache31=$?
if (( status_cache30 != 0 || status_cache31 != 0 )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] cache build failed: F-exp30=${status_cache30}, F-exp31=${status_cache31}" >&2
  exit 1
fi

selected_gpus="$(wait_for_gpus)"
gpus_exp30="$(csv_slice "$selected_gpus" 0 "$NUM_GPUS_PER_EXP")"
gpus_exp31="$(csv_slice "$selected_gpus" "$NUM_GPUS_PER_EXP" "$NUM_GPUS_PER_EXP")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] selected GPUs: ${selected_gpus}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] F-exp30 GPUs: ${gpus_exp30}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] F-exp31 GPUs: ${gpus_exp31}"

run_experiment "$CONFIG_EXP30" "26030" "$RUN_EXP30" "$gpus_exp30" &
pid_exp30=$!
run_experiment "$CONFIG_EXP31" "26031" "$RUN_EXP31" "$gpus_exp31" &
pid_exp31=$!

status_exp30=0
status_exp31=0
wait "$pid_exp30" || status_exp30=$?
wait "$pid_exp31" || status_exp31=$?

if (( status_exp30 != 0 || status_exp31 != 0 )); then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] training failed: F-exp30=${status_exp30}, F-exp31=${status_exp31}" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] both experiments finished; no GPU hold requested"

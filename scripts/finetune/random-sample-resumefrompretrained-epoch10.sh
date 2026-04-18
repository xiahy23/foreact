
set -uo pipefail

WORK_DIR="/media/raid/workspace/xiahongyu/ws_lixing/foreact"
CONFIG="random-sample-resumefrompretrained-epoch10.yaml"
LOG_FILE="${WORK_DIR}/random-sample-resumefrompretrained-epoch10.log"
NUM_GPUS="${1:-8}"
MIN_FREE_MB="${2:-30000}"

cd "$WORK_DIR"

# export HF_HOME=/media/raid/workspace/surongpeng/ws_lixing/huggingface
# export HF_HUB_OFFLINE=1
# export HF_HUB_DOWNLOAD_TIMEOUT=120
# export HF_HUB_ETAG_TIMEOUT=30
export FORCE_VIDEO_BACKEND=pyav

# ---- Helper: check if enough GPUs are free ----
gpus_are_free() {
    local count=0
    while IFS=, read -r idx used total; do
        used=$(echo "$used" | tr -d ' MiB')
        total=$(echo "$total" | tr -d ' MiB')
        free=$((total - used))
        if (( free >= MIN_FREE_MB )); then
            count=$((count + 1))
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    (( count >= NUM_GPUS ))
}

echo "[$(date)] Waiting for ${NUM_GPUS} GPUs with >= ${MIN_FREE_MB} MiB free..."
while ! gpus_are_free; do
    sleep 60
done
echo "[$(date)] GPUs available. Starting training."

# ---- Select GPU IDs with enough free memory ----
GPU_IDS=""
count=0
while IFS=, read -r idx used total; do
    idx=$(echo "$idx" | tr -d ' ')
    used=$(echo "$used" | tr -d ' MiB')
    total=$(echo "$total" | tr -d ' MiB')
    free=$((total - used))
    if (( free >= MIN_FREE_MB )) && (( count < NUM_GPUS )); then
        if [ -n "$GPU_IDS" ]; then GPU_IDS="${GPU_IDS},${idx}"; else GPU_IDS="${idx}"; fi
        count=$((count + 1))
    fi
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "[$(date)] Using GPUs: $CUDA_VISIBLE_DEVICES"

# ---- Launch training ----
echo "[$(date)] Starting F-exp01: config=${CONFIG}, max_steps=-1, gpus=${NUM_GPUS}"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port 29501 \
    --mixed_precision bf16 \
    train.py \
    --config_file "$CONFIG" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Training completed successfully. Log: $LOG_FILE"
else
    echo "[$(date)] Training failed with exit code $EXIT_CODE. See $LOG_FILE."
    exit $EXIT_CODE
fi

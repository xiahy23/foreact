#!/usr/bin/env bash
# =============================================================
# ForeAct finetune: all_eval + bridge_orig_lerobot
# - Waits for GPUs to be free before starting
# - Auto-retries with smaller batch size on OOM
# - Saves every 1000 steps, keeps only 3 latest checkpoints
# =============================================================
set -uo pipefail

WORK_DIR="/media/raid/workspace/xiahongyu/foreact"
CONFIG="finetune_alleval_bridge.yaml"
LOG_FILE="${WORK_DIR}/finetune_alleval_bridge.log"
NUM_GPUS="${1:-8}"
MIN_FREE_MB="${2:-70000}"  # min free GPU memory per card to start

cd "$WORK_DIR"

# Increase HF timeouts (models are cached locally)
export HF_HUB_DOWNLOAD_TIMEOUT=120
export FORCE_VIDEO_BACKEND=pyav
export HF_HUB_ETAG_TIMEOUT=30

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

# ---- Wait for GPUs ----
echo "[$(date)] Waiting for ${NUM_GPUS} GPUs with >= ${MIN_FREE_MB} MiB free..."
while ! gpus_are_free; do
    sleep 60
done
echo "[$(date)] GPUs available. Starting training."

# ---- Select GPU IDs with enough free memory ----
select_gpus() {
    GPU_IDS=""
    local count=0
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
}

select_gpus
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "[$(date)] Using GPUs: $CUDA_VISIBLE_DEVICES"

# ---- Training loop with auto-retry ----
BATCH_SIZES=(4 8 16 32 64)
GRAD_ACCUM=1
MAX_RETRIES=10

for bs_idx in "${!BATCH_SIZES[@]}"; do
    BS="${BATCH_SIZES[$bs_idx]}"
    GA=1

    for retry in $(seq 1 $MAX_RETRIES); do
        echo "[$(date)] Attempt $retry: batch_size=$BS grad_accum=$GA on $NUM_GPUS GPUs..."

        accelerate launch \
            --num_processes "$NUM_GPUS" \
            --mixed_precision bf16 \
            train.py \
            --config_file "$CONFIG" \
            --per_device_train_batch_size "$BS" \
            --gradient_accumulation_steps "$GA" \
            2>&1 | tee -a "$LOG_FILE"

        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[$(date)] Training completed successfully."
            exit 0
        fi

        # Check if it was OOM
        if tail -200 "$LOG_FILE" | grep -qi "out of memory\|CUDA error: out of memory"; then
            echo "[$(date)] OOM detected. Trying smaller batch size..."
            break  # break inner loop to try next batch size
        fi

        # Non-OOM error: wait and retry with same batch size
        echo "[$(date)] Training exited with code $EXIT_CODE (retry $retry/$MAX_RETRIES)."
        sleep 30

        # Re-check GPU availability
        while ! gpus_are_free; do
            echo "[$(date)] Waiting for GPUs..."
            sleep 60
        done
        select_gpus
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    done
done

echo "[$(date)] All attempts exhausted. Check $LOG_FILE for details."
exit 1

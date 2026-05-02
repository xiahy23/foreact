#!/usr/bin/env bash
# =============================================================
# F-exp06: Unfreeze mllm_backbone, 10 epochs (follows F-exp05)
#   - freeze vae only → mllm_backbone is trainable
#   - gradient_checkpointing=true (more params → save memory)
#   - lr=1e-5, constant_with_warmup, bs=8, grad_accum=4
#   - resume_from_checkpoint: checkpoints/F-exp05-default-pretrained-5epoch
#   - 10 epochs, save_strategy=epoch
#   - --main_process_port 25678 to avoid port conflict with F-exp05
#   - Waits for F-exp05 process to finish before launching
#   - Output: checkpoints/F-exp06-unfreeze-mllm-10epoch
# =============================================================
set -uo pipefail

WORK_DIR="/media/raid/workspace/xiahongyu/foreact"
CONFIG="F-exp06-unfreeze-mllm-10epoch.yaml"
LOG_FILE="${WORK_DIR}/F-exp06-unfreeze-mllm-10epoch.log"
NUM_GPUS="${1:-8}"
MASTER_PORT=25678

cd "$WORK_DIR"

export HF_HUB_DOWNLOAD_TIMEOUT=120
export FORCE_VIDEO_BACKEND=pyav
export HF_HUB_ETAG_TIMEOUT=30

# ---- Wait for F-exp05 to finish ----
echo "[$(date)] Waiting for F-exp05 training processes to finish..."
while pgrep -f "F-exp05-default-pretrained-5epoch" > /dev/null 2>&1; do
    sleep 60
done
echo "[$(date)] F-exp05 finished. Sleeping 30s for cleanup..."
sleep 30

# ---- Launch training ----
echo "[$(date)] Starting F-exp06: config=${CONFIG}, unfreeze mllm_backbone, 10 epochs, port=${MASTER_PORT}"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
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

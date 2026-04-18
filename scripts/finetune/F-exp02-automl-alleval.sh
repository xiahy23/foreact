#!/usr/bin/env bash
# =============================================================
# F-exp02: AutoML (Optuna + VLM scoring) for all_eval + bridge
#   Search space (per trial):
#     - per_device_train_batch_size in {16,32,64,128,256,512}
#     - learning_rate (log-uniform), lr_scheduler_type,
#       warmup_steps, unfreeze_mllm
#     - gradient_accumulation_steps FIXED at 1 (no accum)
#   Checkpointing:
#     - save_strategy=epoch, save_total_limit=100
#       (1 ckpt per epoch, all retained per trial)
#   Output:
#     - checkpoints/F-exp02-automl-alleval/trial_XXX/<run_name>
#     - batch_results_F-exp02_automl_alleval/trial_XXX/
#   Resume each trial from: ./foreact-pretrained
# =============================================================
set -o pipefail

WORK_DIR="/media/raid/workspace/xiahongyu/foreact"
LOG_FILE="${WORK_DIR}/F-exp02-automl-alleval.log"
NUM_GPUS="${1:-8}"
N_TRIALS="${2:-12}"
NUM_EPOCHS="${3:-10}"
VLM_MODEL="${4:-doubao-seed-2-0-lite-260215}"
MIN_FREE_MB="${MIN_FREE_MB:-70000}"

cd "$WORK_DIR"

# Conda env
eval "$(conda shell.bash hook)"
conda activate foreact

# Env vars
export FORCE_VIDEO_BACKEND=pyav
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30

if [ -z "${VOLCENKEY:-}" ]; then
    echo "[ERROR] VOLCENKEY not set. Run: export VOLCENKEY='your-api-key'"
    exit 1
fi

# ---- Helper: enough GPUs free? ----
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

echo "[$(date)] Waiting for ${NUM_GPUS} GPUs (>= ${MIN_FREE_MB} MiB free)..."
while ! gpus_are_free; do
    sleep 60
done
echo "[$(date)] GPUs available."

# ---- Select GPU IDs ----
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

# ---- Launch AutoML ----
echo "[$(date)] Starting F-exp02 AutoML: ${N_TRIALS} trials, ${NUM_EPOCHS} epochs, ${NUM_GPUS} GPUs"
python automl_alleval.py \
    --vlm_model "$VLM_MODEL" \
    --n_trials "$N_TRIALS" \
    --num_epochs "$NUM_EPOCHS" \
    --gpus "$NUM_GPUS" \
    2>&1 | tee -a "$LOG_FILE"

echo "[$(date)] F-exp02 finished. See ${LOG_FILE}."

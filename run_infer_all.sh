#!/bin/bash
# Launch inference: multiple workers per GPU

NUM_GPUS=8
PROCS_PER_GPU=3
NUM_WORKERS=$((NUM_GPUS * PROCS_PER_GPU))

PYTHON="/media/raid/workspace/xiahongyu/miniconda3/envs/foreact/bin/python"
SCRIPT_DIR="/media/raid/workspace/xiahongyu/foreact"
CHECKPOINT="./checkpoints/automl_pureorig/trial_000/po_trial_000_lr9.8e-06_bs4_cos_min_ws0/checkpoint-43735"
VIDEO_DIR="./datasets/bridge_orig_lerobot/videos"
COT_PATH="./datasets/cot_by_episode.json"
LOG_DIR="./inference_logs"

cd "$SCRIPT_DIR"
mkdir -p "$LOG_DIR"

echo "Starting inference: $NUM_GPUS GPUs x $PROCS_PER_GPU procs = $NUM_WORKERS workers"
echo "Checkpoint: $CHECKPOINT"

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    for local_id in $(seq 0 $((PROCS_PER_GPU - 1))); do
        worker_id=$((gpu_id * PROCS_PER_GPU + local_id))
        echo "Launching worker $worker_id on GPU $gpu_id..."
        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONUNBUFFERED=1 $PYTHON infer_all_frames.py \
            --checkpoint_path "$CHECKPOINT" \
            --video_dir "$VIDEO_DIR" \
            --cot_path "$COT_PATH" \
            --worker_id $worker_id \
            --num_workers $NUM_WORKERS \
            > "$LOG_DIR/worker_${worker_id}.log" 2>&1 &
        pids+=($!)
    done
done

echo "All $NUM_WORKERS workers launched. PIDs: ${pids[*]}"
echo "Monitor: tail -f $LOG_DIR/worker_*.log"
echo "Waiting..."

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Worker $i finished successfully."
    else
        echo "Worker $i FAILED (exit code $?)."
        failed=$((failed + 1))
    fi
done

if [ $failed -eq 0 ]; then
    echo "All workers finished successfully!"
else
    echo "$failed worker(s) failed. Check logs in $LOG_DIR/"
fi

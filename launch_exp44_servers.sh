#!/bin/bash
# Launch 5 foreact servers for exp44 checkpoints
# checkpoint-400/800/1200/1600/2000 -> GPU 0-4, port 5100-5104

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate foreact

BASE_CKPT="checkpoints/F-exp44-three_object_foreact_from_F-exp31_s1_t2s_bs64x4_e50_nofilter"
STEPS=(400 800 1200 1600 2000)
BASE_PORT=5100

mkdir -p logs

PIDS=()

for i in "${!STEPS[@]}"; do
    STEP=${STEPS[$i]}
    PORT=$((BASE_PORT + i))
    GPU=$i
    CKPT_PATH="${BASE_CKPT}/checkpoint-${STEP}"
    LOG="logs/server_ckpt${STEP}.log"

    echo "Starting: checkpoint-${STEP} | GPU ${GPU} | port ${PORT}"
    CUDA_VISIBLE_DEVICES=${GPU} python server_foreact.py \
        --checkpoint_path "${CKPT_PATH}" \
        --port ${PORT} \
        --idle_timeout -1 \
        > "${LOG}" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "  PID=${PID}  log=${LOG}"
done

echo ""
echo "All 5 servers launched. PIDs: ${PIDS[*]}"
echo "${PIDS[*]}" > logs/server_pids.txt
echo ""
echo "Wait for models to load (~1-2 min), then run:"
echo "  python query_exp44_servers.py --image <your_image.png> --task 'your task'"
echo ""
echo "Check logs:  tail -f logs/server_ckpt*.log"
echo "Stop all:    bash stop_exp44_servers.sh"

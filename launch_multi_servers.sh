#!/bin/bash
# Launch 5 foreact servers for F-exp44 checkpoints, one per port/GPU.
# Usage: bash launch_multi_servers.sh
# Logs go to /tmp/foreact_server_<ckpt>.log

BASE_DIR="/media/raid/workspace/xiahongyu/foreact"
CKPT_BASE="${BASE_DIR}/checkpoints/F-exp44-three_object_foreact_from_F-exp31_s1_t2s_bs64x4_e50_nofilter"

CHECKPOINTS=(
    "checkpoint-400"
    "checkpoint-800"
    "checkpoint-1200"
    "checkpoint-1600"
    "checkpoint-2000"
)

# Ports 5140~5144; adjust CUDA_VISIBLE_DEVICES as needed
PORTS=(5140 5141 5142 5143 5144)
GPUS=(0 1 2 3 4)

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    PORT="${PORTS[$i]}"
    GPU="${GPUS[$i]}"
    LOG="/tmp/foreact_server_${CKPT}.log"

    echo "Starting server for ${CKPT} on port ${PORT} (GPU ${GPU}) ..."
    CUDA_VISIBLE_DEVICES=${GPU} python "${BASE_DIR}/server_foreact.py" \
        --checkpoint_path "${CKPT_BASE}/${CKPT}" \
        --port ${PORT} \
        --idle_timeout -1 \
        > "${LOG}" 2>&1 &

    echo "  PID=$! log=${LOG}"
done

echo ""
echo "All 5 servers launched. Ports: ${PORTS[*]}"
echo "Wait ~30s for models to load, then run client_multi_ckpt.py"

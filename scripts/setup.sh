#!/bin/bash

DEFAULT_GPUS_PER_NODE=4
DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=25002

echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"

JOB_NAME=${JOB_NAME:-$DEFAULT_JOB_NAME}
JOB_NAME=${JOB_NAME:-$SLURM_JOB_NAME}
echo "JOB_NAME = $JOB_NAME"

NNODES=${SLURM_JOB_NUM_NODES:-1}
echo "NNODES = $NNODES"

if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_NODELIST" ]; then
    NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
else
    NODES="localhost"
fi
echo "NODES = $NODES"

NODE_RANK=${SLURM_PROCID:-0}
echo "NODE_RANK = $NODE_RANK"

GPUS_PER_NODE=${SLURM_JOB_GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
echo "GPUS_PER_NODE = $GPUS_PER_NODE"

if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_NODELIST" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
    MASTER_ADDR=$DEFAULT_MASTER_ADDR
fi
echo "MASTER_ADDR = $MASTER_ADDR"

MASTER_PORT=${MASTER_PORT:-$DEFAULT_MASTER_PORT}
echo "MASTER_PORT = $MASTER_PORT"

TORCHRUN_ARGS="--nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT"

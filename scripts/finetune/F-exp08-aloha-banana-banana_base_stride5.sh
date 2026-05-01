#!/usr/bin/env bash
set -euo pipefail
cd /media/raid/workspace/xiahongyu/foreact
export FORCE_VIDEO_BACKEND=pyav
export WANDB_MODE=offline
NUM_GPUS="${1:-8}"
conda run --no-capture-output -n foreact accelerate launch --num_processes "$NUM_GPUS" --main_process_port 26008 --mixed_precision bf16 train.py --config_file "F-exp08-aloha-banana-banana_base_stride5.yaml"

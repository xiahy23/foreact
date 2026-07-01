#!/usr/bin/env bash
set -euo pipefail
cd /media/raid/workspace/xiahongyu/foreact
export FORCE_VIDEO_BACKEND=pyav
export WANDB_MODE=offline
NUM_GPUS="${NUM_GPUS:-4}"
conda run --no-capture-output -n foreact accelerate launch --num_processes "$NUM_GPUS" --main_process_port 26030 --mixed_precision bf16 train.py --config_file "F-exp28-obstacle-left-first_motion_delta005_s5_t2s.yaml"

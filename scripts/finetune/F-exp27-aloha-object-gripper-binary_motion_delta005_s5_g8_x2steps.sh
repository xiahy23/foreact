#!/usr/bin/env bash
set -euo pipefail
cd /media/raid/workspace/xiahongyu/foreact
export FORCE_VIDEO_BACKEND=pyav
export WANDB_MODE=offline
conda run --no-capture-output -n foreact accelerate launch --num_processes 8 --main_process_port 26029 --mixed_precision bf16 train.py --config_file "F-exp27-aloha-object-gripper-binary_motion_delta005_s5_g8_x2steps.yaml"

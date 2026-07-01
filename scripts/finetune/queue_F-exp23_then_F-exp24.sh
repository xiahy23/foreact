#!/usr/bin/env bash
set -euo pipefail

cd /media/raid/workspace/xiahongyu/foreact
exec scripts/finetune/queue_F-exp23_F-exp24_parallel_then_act_cot.sh "$@"

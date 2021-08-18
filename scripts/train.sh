#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
log_name="LOG_$2_$3_$now"
CUDA_VISIBLE_DEVICES=$1 python3 -u main.py --benchmark $2 --exp_name $3 ${@:4} 2>&1|tee logs/$log_name.log

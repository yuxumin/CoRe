#!/usr/bin/env sh
mkdir -p logs_test
now=$(date +"%m%d_%H%M")
log_name="LOG_$2_$3_$now"
CUDA_VISIBLE_DEVICES=$1 python3 -u main.py --benchmark $2 --exp_name $3 --test ${@:4} 2>&1|tee logs_test/$log_name.log

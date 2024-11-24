#!/bin/bash
export PYTHONPATH=$(realpath .):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

export MASTER_ADDR=localhost
export MASTER_PORT=6001

python data_engine/data_engine.py \
      --reward_model_name llava-v1.5-7b \
      --reward_model_path /data/yaoshu/models/llava-v1.5-7b \
      --instruct_model_name RLAIF-V-7B \
      --instruct_model_path /data/yaoshu/models/RLAIF-V-7B \
      --dataset_path /data/yaoshu/dataset/RLAIF-V-Dataset \
      --work_dir /data/RLAIF-V-CC/results/test \
      --continue_from_stage 2
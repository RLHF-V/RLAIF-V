#!/bin/bash
export PYTHONPATH=$(realpath .):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=6001
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS data_engine/data_engine.py \
      --reward_model_name RLAIF-V-7B \
      --reward_model_path /data/yaoshu/models/RLAIF-V-7B \
      --instruct_model_name llava-v1.5-7b \
      --instruct_model_path /data/yaoshu/models/llava-v1.5-7b \
      --dataset_path /data/yaoshu/dataset/origin_dataset \
      --work_dir /data/RLAIF-V-CC/results/test1 \
      --image_column image_bytes \
      --continue_from_stage 0
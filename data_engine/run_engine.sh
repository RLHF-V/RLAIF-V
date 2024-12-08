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

reward_model_name="reward_model_name"
reward_model_path="/path/to/yout/reward/model"
instruct_model_name="instruct_model_name"
instruct_model_path="/path/to/yout/instruct/model"
dataset_path="/path/to/your/dataset"
work_dir="/path/to/your/work/dir"
pipeline_name="pipeline_name"
reward_model_python_path="needed_for_minicpmv"


torchrun $DISTRIBUTED_ARGS data_engine/engine.py \
      --reward_model_name $reward_model_name \
      --reward_model_path $reward_model_path \
      --instruct_model_name $instruct_model_name \
      --instruct_model_path $instruct_model_path \
      --dataset_path $dataset_path \
      --work_dir  $work_dir \
      --pipeline_name $pipeline_name \
      --reward_model_python_path $reward_model_python_path \
      --run_stage 0\
      --debug

torchrun $DISTRIBUTED_ARGS data_engine/engine.py \
      --reward_model_name $reward_model_name \
      --reward_model_path $reward_model_path \
      --instruct_model_name $instruct_model_name \
      --instruct_model_path $instruct_model_path \
      --dataset_path $dataset_path \
      --work_dir  $work_dir \
      --pipeline_name $pipeline_name \
      --reward_model_python_path $reward_model_python_path \
      --run_stage 1\
      --debug

torchrun $DISTRIBUTED_ARGS data_engine/engine.py \
      --reward_model_name $reward_model_name \
      --reward_model_path $reward_model_path \
      --instruct_model_name $instruct_model_name \
      --instruct_model_path $instruct_model_path \
      --dataset_path $dataset_path \
      --work_dir  $work_dir \
      --pipeline_name $pipeline_name \
      --reward_model_python_path $reward_model_python_path \
      --run_stage 2\
      --debug
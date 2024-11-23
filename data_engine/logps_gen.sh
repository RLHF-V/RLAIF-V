#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`realpath .`
export CUDA_VISIBLE_DEVICES=1

#GPUS_PER_NODE=1
#NNODES=1
#NODE_RANK=0
export MASTER_ADDR=localhost
#MASTER_ADDR=13.13.19.1
export MASTER_PORT=6001
#
#DISTRIBUTED_ARGS="
#    --nproc_per_node $GPUS_PER_NODE \
#    --nnodes $NNODES \
#    --node_rank $NODE_RANK \
#    --master_addr $MASTER_ADDR \
#    --master_port $MASTER_PORT
#"

#torchrun $DISTRIBUTED_ARGS data_engine/logps_calculator.py  \
#          --use_12b_model_for_reward \
#          --use_12b_model_for_instruct

python data_engine/logps_calculator.py
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`realpath .`

echo Working Directory at `pwd`
echo Bash at `which bash`
echo Python at `which python`

ckpt=$1

ans_dir=$2

ques_dir=$3
ques_file=$4

echo "question dir: "$ques_dir
echo "question file: "$ques_file
echo "answer dir: "$ans_dir

start_pos=$5
end_pos=$6

num_gpus=$7

echo "start pos "$start_pos" end pos "$end_pos

torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 ./omnilmm/omnilmm_gen_data.py \
--checkpoint $ckpt \
--ds_name ${ques_dir}/${ques_file} \
--answer_file ${ans_dir}/autocheck_omni_${start_pos}-${end_pos}_${ques_file} \
--max_sample -1 \
--start_pos $start_pos \
--end_pos $end_pos \
--max_tokens 1 \
--num-workers 5 \
--batch-size 1 \
--is_yesno
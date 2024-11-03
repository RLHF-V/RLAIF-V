#!/bin/bash

# `model_data_dir` is the directory for saving evaluation results
# `path_model` is the file path for the model answers
# `model_name` is the evaluated model name
path_gpt4v="eval/data/gpt4v_RefoMB_dev_0521.jsonl"
model_data_dir=$1
path_model=$2
model_name=$3

echo $path_gpt4v

python eval/eval_RefoMB_p0.py \
    --answer_gpt_4v  $path_gpt4v \
    --answer_model $path_model \
    --save_dir $model_data_dir \
    --modelA GPT-4V \
    --modelB $model_name

python eval/json_to_excel.py --text_prompt "${model_data_dir}/A-GPT-4V_B-${model_name}.json" --get_all_data

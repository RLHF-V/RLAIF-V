#!/bin/bash

# `file_path` is the overall score evaluation result file path, without ".json", e.g. "./results/A-GPT-4V_B-rlaifv7b"
# hallucination evaluation result will be save at "${file_path}.hall.json"

file_path=$1
save_path=${file_path}.hall
python eval/eval_hallucination.py \
    --jsonl_file ${file_path}.json

python eval/json_to_excel.py --text_prompt ${save_path}.json

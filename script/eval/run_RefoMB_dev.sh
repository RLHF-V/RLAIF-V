#!/bin/bash

python ./RefoMB/eval_RefoMB_dev.py \
    --answer_modelA    ./eval/data/RefoMB_gpt4v_dev.jsonl \
    --answer_modelB    /home/dangyunkai/openmme_0511_eval/baseline/Mini-Gemini-34B_openmme_val.jsonl

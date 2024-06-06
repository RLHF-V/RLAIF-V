# An example of generate - divide and conquer - feedback generation using OmniLMM 12B

### diverse gen
model_path=/home/yutianyu/Zephyr_checkpoints/SFT_exp/zephyr_mm_12b_SFT-knowledge_v0-caterpillar-stage2_mix#caterpillar-stage3_lvis#caterpillar-stage3_svit#caterpillar-stage3_sharegpt4v#llava#unimm-chat#mme_art500k#google_landmark-train#places_365-train#wit-en-13#22#60#10#16#12#15#10#2#20/checkpionts/checkpoint-4000
ques_dir=/home/zhanghaoye/apps/DPO/data/diversify/0506_omni_base_scaling/2iter_data/ # directory of the input file
ques_file=0313_division_detailed_input # name of the input file, without '.jsonl'
ans_dir=/home/zhanghaoye/apps/DPO/data/diversify/0506_omni_base_scaling/2iter_data/omni12b_base_detail # directory of the answer files
start=0
end=-1
bash ./script/data_gen/omnilmm/omnilmm_diverse_gen.sh \
$model_path \
$ans_dir \
$ques_dir \
${ques_file}.jsonl \
$start \
$end

### llama3 divide
ans_file=${ans_dir}/diverse_gen_${start}-${end}_${ques_file} # path of the generation result file
bash ./script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh \
$ans_file \
0 \
-1 \
8

## autocheck
model_path=/home/yutianyu/Zephyr_checkpoints/SFT_exp/zephyr_mm_12b_SFT-knowledge_v0-caterpillar-stage2_mix#caterpillar-stage3_lvis#caterpillar-stage3_svit#caterpillar-stage3_sharegpt4v#llava#unimm-chat#mme_art500k#google_landmark-train#places_365-train#wit-en-13#22#60#10#16#12#15#10#2#20/checkpionts/checkpoint-4000
check_ques_file=diverse_gen_${start}-${end}_${ques_file}.s0-e-1.llama3-8b_divide.gq.qas.jsonl # name of the divide-and-conquer result file

bash ./script/data_gen/omnilmm/omnilmm_autocheck.sh \
$model_path \
$ans_dir \
$ans_dir \
$check_ques_file \
0 \
-1
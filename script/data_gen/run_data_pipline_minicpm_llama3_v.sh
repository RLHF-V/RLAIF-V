# An example of generate - divide and conquer - feedback generation using MiniCPM-Llama3-V 2.5

### diverse gen
model_path=/home/zhanghaoye/MiniCPMV_checkpoints_zhy/DPO_exp/minicpmv_llama3_multilingual/minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1/checkpoints/checkpoint-280
ans_dir=/home/zhanghaoye/apps/DPO/data/diversify/0518_iter_data_sampled_minicpmv_llama3_multilingual/2iter/minicpmv_llama3_gen_detail
ques_dir=/home/zhanghaoye/apps/DPO/data/diversify/0518_iter_data_sampled_minicpmv_llama3_multilingual/2iter/
ques_file=0313_division_detailed_input_1800
start=0
end=-1

bash ./script/data_gen/minicpm_llama3_v/minicpm_llama3_v_diverse_gen.sh \
$model_path \
$ans_dir \
$ques_dir \
${ques_file}.jsonl \
$start \
$end \
8

### llama3 divide
ans_file=${ans_dir}/diverse_gen_${start}-${end}_${ques_file} # path of the generation result file
bash ./script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh \
$ans_file \
0 \
-1 \
8

### autocheck qa
check_ques_file=diverse_gen_minicpmv25.s${start}-e${end}.${ques_file}.s0-e-1.llama3-8b_divide.gq.qas.jsonl

bash ./script/data_gen/minicpm_llama3_v/minicpm_llama3_v_autocheck.sh \
$model_path \
$ans_dir \
$ans_dir \
$check_ques_file \
0 \
-1 \
8


base_dir="./LLaVA15_7B_checkpoints/SelfReward_exp/llava15_7b_DPO-sr_llava15_llava15base_rmllava16_data_1iter_eq2000imgs-sr_llava15_llava15base_rmllava16_data_base_eq2000imgs-1/checkpoints"

echo "model dir: "$base_dir

review_file_name=hall_1025_ver_chair_-1.json

python ./script/eval/eval_gpt_chair_modelbest.py \
    --cap_folder $base_dir \
    --cap_type 1025_ver_chair.jsonl \
    --use_gpt

python ./script/eval/summarize_chair.py $base_dir $review_file_name > $base_dir/summ_chair.txt

review_file_name=hall_1025_ver_chair_-1_mapping0406.json

python ./script/eval/eval_gpt_chair_modelbest.py \
    --cap_folder $base_dir \
    --cap_type 1025_ver_chair.jsonl \
    --use_gpt \
    --do_rescore

python ./script/eval/summarize_chair.py $base_dir $review_file_name > $base_dir/summ_chair_0406mapping.txt


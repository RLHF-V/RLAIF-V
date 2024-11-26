export PYTHONPATH=$PYTHONPATH:`realpath .`

task_name=llava15_7b_DPO
exp_name=llava15_rlaifv_lora

deepspeed ./muffin/train/train_llava15_lora.py \
    --deepspeed ./script/zero2.json  \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --data_dir ./RLAIF-V-Dataset_logps/ \
    --image_folder not_used \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune False \
    --image_aspect_ratio pad \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --output_dir .ckpt/$task_name-$exp_name/checkpoints \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 167 \
    --save_total_limit 50 \
    --data_source_names '' \
    --data_source_weights 1 \
    --max_steps 2672 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --logging_dir .ckpt/$task_name-$exp_name/log \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $exp_name \
    --dataloader_num_workers 16 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta 0.1 \
    --lora_enable True

parent_dir=".ckpt/$task_name-$exp_name/checkpoints"
config_file="$parent_dir/config.json"
non_lora_trainables="$parent_dir/non_lora_trainables.bin"

for dir in "$parent_dir"/checkpoint-*; do
    if [ -d "$dir" ]; then
        base_name=$(basename "$dir")
        new_name=${base_name/checkpoint-/RLAIFV7B-lora_checkpoint-}
        new_dir="$parent_dir/$new_name"

        mv "$dir" "$new_dir"
        echo "Renamed $dir to $new_dir"

        cp "$config_file" "$new_dir/"
        cp "$non_lora_trainables" "$new_dir/"
        echo "Copied $config_file to $new_dir/"
        echo "Copied $non_lora_trainables to $new_dir/"
    fi
done


task_name=llava15_7b_DPO
exp_name=sr_llava15_llava15base_rmllava16_data_1iter_eq4000imgs
sft_data=sr_llava15_llava15base_rmllava16_data_base_eq4000imgs

/home/jeeves/miniconda3/envs/rlaifv/bin/deepspeed ./muffin/train/train_llava15.py \
    --deepspeed ./script/zero3.json  \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --image_folder not_used \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fully_tune True \
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
    --data_source_names  $sft_data \
    --data_source_weights 1 \
    --max_steps 2672 \
    --learning_rate 5e-7 \
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
    # --eval_steps 50 \
    # --eval_data_source_names RM_Bench_clean_diff1#RM_Bench_clean_diff2#RM_Bench_clean_diff3

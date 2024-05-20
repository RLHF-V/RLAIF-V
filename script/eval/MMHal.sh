
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
CKPT=llava15_1iter

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./muffin/eval/llava15_chair.py \
        --model-path /home/luxiaoman/RLAIF_V/LLaVA15_7B_checkpoints/SelfReward_exp/llava15_7b_DPO-sr_llava15_llava15base_rmllava16_data_1iter_eq4000imgs-sr_llava15_llava15base_rmllava16_data_base_eq4000imgs-1/checkpoints/checkpoint-1002 \
        --question-file ./data/eval/MMHalBench/response_template_qa700.json \
        --answers-file ./data/eval/MMHalBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 &
done

wait

output_file=./data/eval/MMHalBench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/MMHalBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done





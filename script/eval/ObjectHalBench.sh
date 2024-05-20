
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ./muffin/eval/llava15_chair.py \
        --model-path liuhaotian/llava1.5-7b \
        --question-file ./data/eval/ObjectHalBench/chair_instruction2_img_diversify_8_noemo_test_300.jsonl \
        --answers-file ./data/eval/ObjectHalBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --num_beams 3 &
done

wait

output_file=./data/eval/ObjectHalBench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./data/eval/ObjectHalBench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done





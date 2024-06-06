echo Working Directory at `pwd`
echo Bash at `which bash`
echo Python at `which python`

echo `hostname`

data_path=$1
start=$2
end=$3
bs=$4

chunk_num=$5
echo "data_path="$data_path
echo "chunk_num="$chunk_num
echo "start="$start" end="$end
echo "batch_size="$bs


for i in $(seq 0 $((chunk_num-1))); do
    echo $i
    CUDA_VISIBLE_DEVICES=$i python ./utils/llama3_8b_inference.py \
    --path ${data_path}.jsonl \
    --chunk-num $chunk_num \
    --chunk-idx $i \
    --bs $bs \
    --start ${start} \
    --end ${end} &
done
wait

# Merge divided files
output_file=${data_path}.s${start}-e${end}.llama3-8b_divide.jsonl
> "$output_file"
for IDX in $(seq 0 $((chunk_num-1))); do
    cat ${data_path}.s${start}-e${end}.chunk${chunk_num}-${IDX}.llama3-8b_divide.jsonl >> "$output_file"
done

# Merge generated questions files
output_file=${data_path}.s${start}-e${end}.llama3-8b_divide.gq.jsonl
> "$output_file"
for IDX in $(seq 0 $((chunk_num-1))); do
    cat ${data_path}.s${start}-e${end}.chunk${chunk_num}-${IDX}.llama3-8b_divide.gq.jsonl >> "$output_file"
done

# Merge generated questions with 'yes or no' suffix files
output_file=${data_path}.s${start}-e${end}.llama3-8b_divide.gq.qas.jsonl
> "$output_file"
for IDX in $(seq 0 $((chunk_num-1))); do
    cat ${data_path}.s${start}-e${end}.chunk${chunk_num}-${IDX}.llama3-8b_divide.gq.qas.jsonl >> "$output_file"
done

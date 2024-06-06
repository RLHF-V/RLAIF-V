export PYTHONPATH=$PYTHONPATH:`realpath .`

echo Working Directory at `pwd`
echo Bash at `which bash`

ckpt=$1

ans_dir=$2

ques_dir=$3
ques_file=$4

echo "question dir: "$ques_dir
echo "question file: "$ques_file
echo "answer dir: "$ans_dir

start_pos=$5
end_pos=$6

chunk_num=$8

echo "start pos "$start_pos" end pos "$end_pos

PYTHON_ENV=$7
echo Python at ${PYTHON_ENV}

for i in $(seq 0 $((chunk_num-1))); do
    echo ${chunk_num}-${i}
    CUDA_VISIBLE_DEVICES=$i ${PYTHON_ENV} ./minicpm-llama3-v-25/minicpmv_diverse_gen.py \
    --model-name $ckpt \
    --question-file ${ques_dir}/${ques_file} \
    --answers-file ${ans_dir}/diverse_gen_minicpmv25.s${start_pos}-e${end_pos}.chunk${chunk_num}-${i}.${ques_file} \
    --start $start_pos \
    --end $end_pos \
    --repeat 10 \
    --chunk-num ${chunk_num} \
    --chunk-idx ${i} \
    --sampling &
done
wait

output_file=${ans_dir}/diverse_gen_minicpmv25.s${start_pos}-e${end_pos}.${ques_file}

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((chunk_num-1))); do
    cat ${ans_dir}/diverse_gen_minicpmv25.s${start_pos}-e${end_pos}.chunk${chunk_num}-${IDX}.${ques_file} >> "$output_file"
done

#!/bin/bash

#SBATCH --partition=gpu3-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=g3030
#SBATCH --output=./_temp/slurm_output/%j.%x.out

export PYTHONPATH=$PYTHONPATH:`realpath .`

base_dir=""
to_process_ckpt_list=`ls -d1 $base_dir*/*/checkpionts/checkpoint-*`
to_process_ckpt_list+=`ls -d1 $base_dir*/checkpionts/checkpoint-*`
to_process_ckpt_list+=`ls -d1 $base_dir*/*/checkpoints/checkpoint-*`
to_process_ckpt_list+=`ls -d1 $base_dir*/checkpoints/checkpoint-*`


# =========> chair <============
tag=""
answer_file_name=1025_ver_chair$tag.jsonl

filered_to_process_ckpt_list=""
for ckpt_path in $to_process_ckpt_list;
do
    [[ ! -d $ckpt_path ]] && continue

    if [[ ! -f $ckpt_path/$answer_file_name ]]; then
        echo EVAL $ckpt_path/$answer_file_name
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt_path
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt_path
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"

C=0
QUES_FILE=chair_instruction2_img_diversify_8_noemo_test_300

for ckpt_path in $filered_to_process_ckpt_list;
do
    echo "PWD at `pwd` checkpoint: "$ckpt_path/$answer_file_name" do CHAIR"
    save_dir=$ckpt_path
    CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/llava15_chair.py \
        --model-path $ckpt_path \
        --question-file ./script/eval/data/$QUES_FILE.jsonl \
        --answers-file $save_dir/$answer_file_name \
        --temperature 0 \
        --num_beams 3 &
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi

done
wait

# =========> mm-hal bench <============

answer_file_name="mmhal-bench_answer$tag.jsonl"
filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $ckpt/$answer_file_name
    if [[ ! -f $ckpt/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"


C=0
q_file=script/eval/data/response_template_qa700.json

for ckpt_path in $filered_to_process_ckpt_list;
do
    save_dir=$ckpt_path
    answer_file=$save_dir/$answer_file_name

    echo "PWD at `pwd` checkpoint: "$ckpt_path" output to: "$answer_file

    CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/llava15_chair.py \
        --model-path $ckpt_path \
        --question-file $q_file \
        --answers-file $answer_file \
        --temperature 0 \
        --num_beams 3 &
    C=$((C+1))
    echo "C=$C"
    if [[ $C == 8 ]]; then
        echo "Wait for next iteration"
        C=0
        wait
    fi
done
wait

echo "========>Done generating answers<========"


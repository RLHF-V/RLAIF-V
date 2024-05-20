###===> install dependencies
export PYTHONPATH=$PYTHONPATH:`realpath .`
export TORCH_DISTRIBUTED_DEBUG=DETAIL
echo "pythonpath="$PYTHONPATH
###<===

base_dir=$1
to_process_ckpt_list="$1 "
save_dir=$2


q_file=./eval/data/mmhal-bench_with_image.jsonl
template_file=./eval/data/mmhal-bench_answer_template.json
answer_file_name=mmhal-bench_answer.jsonl

filered_to_process_ckpt_list=""
for ckpt in $to_process_ckpt_list;
do
    [[ ! -d $ckpt ]] && continue

    echo $save_dir/$answer_file_name
    if [[ ! -f $save_dir/$answer_file_name ]]; then
        filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
    fi
    # filered_to_process_ckpt_list=$filered_to_process_ckpt_list" "$ckpt
done
echo "Process these checkpoints: [$filered_to_process_ckpt_list]"

C=3

for ckpt_path in $filered_to_process_ckpt_list;
do
    if [[ ! -f $save_dir/$answer_file_name ]]; then
        answer_file=$save_dir/$answer_file_name
        echo "PWD at `pwd` checkpoint: "$ckpt_path" do MMHal-Bench output to: "$answer_file

        CUDA_VISIBLE_DEVICES=$C python ./muffin/eval/muffin_vqa.py \
            --model-path $ckpt_path \
            --question-file $q_file \
            --answers-file $answer_file &
        C=$((C+1))
        echo "C=$C"
        if [[ $C == 8 ]]; then
            echo "Wait for next iteration"
            C=0
            wait
        fi
    fi

done
wait
echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"

answer_file=$save_dir/$answer_file_name

python ./eval/change_mmhal_predict_template.py \
    --response-template $template_file \
    --answers-file $answer_file \
    --save-file $answer_file.template.json

python ./eval/eval_gpt_mmhal.py \
    --response $answer_file.template.json \
    --evaluation $answer_file.mmhal_test_eval.json \
    --api-key $3 >> ${answer_file}.eval_log.txt

# Merge gpt4 evaluation to the original model outputs, can be ignore
python ./eval/merge_mmhal_review_with_predict.py \
    --review_path ${answer_file}.mmhal_test_eval.json \
    --predict_path ${answer_file} \
    --save_path ${answer_file}.mmhal_test_all_infos.json

python ./eval/summarize_gpt_mmhal_review.py $save_dir > $save_dir/mmhal_scores.txt

# Print Log
echo Scores are:
cat $save_dir/mmhal_scores.txt
echo done
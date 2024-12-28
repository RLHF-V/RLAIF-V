ckpt=$1

ans_dir=$2

ques_dir=$3

echo "question dir: "$ques_dir
echo "answer dir: "$ans_dir

PYTHON_ENV=$4

num_gpus=$5

source /opt/conda/bin/activate $PYTHON_ENV
#export PYTHONPATH=PYTHON_ENV
echo Working Directory at `pwd`
echo Bash at `which bash`
echo Python at `which python`

torchrun --nnodes=1 --nproc_per_node=${num_gpus} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 ./muffin/llava_critic_gen_data.py \
--checkpoint $ckpt \
--ds_name ${ques_dir} \
--answer_dir ${ans_dir} \
--max_tokens 512 \
--num-workers 5 \
--batch-size 4 \
--temperature 0.7
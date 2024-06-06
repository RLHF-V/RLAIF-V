export PYTHONPATH=$PYTHONPATH:`realpath .`

echo Working Directory at `pwd`
echo Bash at `which bash`
echo Python at `which python`

python ./utils/get_preference_pairs.py \
--autocheck_path $1 \
--gpt_divide_gq_path $2 \
--sample_num $3
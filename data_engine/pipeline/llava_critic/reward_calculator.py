import os
import glob
from copy import deepcopy
import subprocess

import pandas as pd
import torch

from data_engine.util import dir_prepare
from llava.constants import DEFAULT_IMAGE_TOKEN

__critic_prompt = DEFAULT_IMAGE_TOKEN + "\n" + "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [{}]\nThe first response: [{}]\nThe second response: [{}]\nASSISTANT:\n"


def __eval_builder(directory: str, store_dir: str):
    """
    Processes all .parquet files in the given directory, groups the data by 'idx',
    and builds critic prompts for each pair of responses within the same group.

    Args:
        directory (str): Path to the directory containing .parquet files.
        store_dir (str): Path to store processed prompts
    """
    parquet_files = glob.glob(os.path.join(directory, "*.parquet"))
    if not parquet_files:
        raise ValueError(f"No .parquet files found in directory: {directory}")

    # Initialize an empty list to collect all data
    all_data = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            data = df.to_dict(orient='records')
            all_data.extend(data)
            print(f"Loaded {len(data)} records from {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        raise ValueError("No data loaded from the .parquet files.")

    answers = deepcopy(all_data)

    grouped_data = {}
    for item in answers:
        idx = item.get('idx')
        if idx is None:
            raise KeyError("Each data item must have an 'idx' field.")
        if idx not in grouped_data:
            grouped_data[idx] = []
        grouped_data[idx].append(item)

    answer_pairs = list(grouped_data.values())

    print(f"Grouped data into {len(answer_pairs)} groups based on 'idx' field.")

    res = []
    for pairs in answer_pairs:
        for index, pair in enumerate(pairs):
            pair_inner_idx = index
            pair["inner_idx"] = pair_inner_idx

            question = pair.get("question")
            response_1 = pair.get("chosen")

            if question is None or response_1 is None:
                raise KeyError("Each data item must have 'question' and 'chosen' fields.")

            for compare_index, compare in enumerate(pairs):
                if compare_index == index:
                    # print("equal")
                    continue
                response_2 = compare.get("chosen")
                if response_2 is None:
                    raise KeyError("Each data item must have a 'chosen' field for comparison.")

                prompt = __critic_prompt.format(question, response_1, response_2)
                pair["question"] = prompt
                res.append(deepcopy(pair))

    print("Completed building critic prompts for all pairs.")

    step = 5000
    print(f"got {len(res)}, storing ...")
    for idx, start in enumerate(range(0, len(res), step)):
        temp_data = res[start: min(start + step, len(res))]
        df = pd.DataFrame(temp_data)
        df.to_parquet(
            os.path.join(store_dir, f'RLAIF-V-Dataset-llava_critic-prompts_{idx:03}-{len(temp_data)}.parquet'))


def __run_bash_script(script_path, *args):
    """Helper function to run bash scripts with arguments."""
    command = ['bash', script_path] + list(args)
    subprocess.run(command, check=True)


def calculate_reward(sampled_answer_path: str, work_dir: str, model_path: str, llava_critic_python_path:str):
    if torch.distributed.get_rank() == 0:
        prompts = os.path.join(work_dir, "prompts")
        dir_prepare(prompts)
        __eval_builder(sampled_answer_path, prompts)

        res_dir = os.path.join(work_dir, "res")
        dir_prepare(res_dir)
        script_path = './script/data_gen/llava_critic/llava_critic_gen.sh'
        __run_bash_script(
            script_path,
            model_path,
            res_dir,
            prompts,
            llava_critic_python_path,
            str(torch.cuda.device_count()))

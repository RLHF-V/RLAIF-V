import os
from copy import deepcopy
import subprocess
import warnings

import pandas as pd
import torch
from tqdm import tqdm

from data_engine.util import dir_prepare
from llava.constants import DEFAULT_IMAGE_TOKEN
from data_engine.util import read_parquets


def __run_bash_script(script_path, *args):
    command = ['bash', script_path] + list(args)
    subprocess.run(command, check=True)


better = ["[first]", "The first response is better"]
worse = ["[second]", "The second response is better"]
equal = ["Two responses are equally", "The two responses are equally good", "The two responses are identical", "Both responses are equally good"]


def __check_contains(list_to_check, response):
    for item in list_to_check:
        if item in response:
            return True
    return False


def __extract_and_calculate(critic_res_dir: str, sampled_answer_path: str):
    critic_res = pd.DataFrame(read_parquets(critic_res_dir))
    sampled_answer = pd.DataFrame(read_parquets(sampled_answer_path))

    scores = []
    for _, item in tqdm(sampled_answer.iterrows(), total=sampled_answer.shape[0], desc="Calculating scores"):
        idx = item.get('idx')
        inner_idx = item.get('inner_idx')

        critic_res_items = critic_res.query(f'idx == "{idx}" and inner_idx == {inner_idx}')
        item = deepcopy(item)
        score = 0
        for _, row in critic_res_items.iterrows():
            critic = row.get('answer')
            if __check_contains(better, critic):
                score += 1
            elif __check_contains(worse, critic):
                score -= 1
            elif __check_contains(equal, critic):
                score += 0
            else:
                warnings.warn(f"Invalid critic response: \n{critic}")
                # raise ValueError(f"Invalid critic response: \n{critic}")
        item['score'] = score
        scores.append(item.to_dict())
    return scores


def calculate_reward(sampled_answer_path: str,
                     work_dir: str,
                     model_path: str,
                     llava_critic_python_path: str,
                     reward_path: str):
    if torch.distributed.get_rank() == 0:
        res_dir = os.path.join(work_dir, "res")
        dir_prepare(res_dir)
        script_path = './script/data_gen/llava_critic/llava_critic_gen.sh'
        __run_bash_script(
            script_path,
            model_path,
            res_dir,
            sampled_answer_path,
            llava_critic_python_path,
            str(torch.cuda.device_count()))

        scores = __extract_and_calculate(res_dir, sampled_answer_path)

        step = 5000
        for idx, start in enumerate(range(0, len(scores), step)):
            temp_data = scores[start: min(start + step, len(scores))]
            # Verify data before creating DataFrame
            if len(temp_data) == 0:
                continue  # Skip empty batches

            df = pd.DataFrame(temp_data)
            output_file = os.path.join(
                reward_path,
                f'RLAIF-V-Dataset-scores_{idx:03}-{len(temp_data)}.parquet'
            )
            temp_file = output_file + '.tmp'
            df.to_parquet(temp_file)
            os.rename(temp_file, output_file)
            print(f"Saved {len(temp_data)} records to {output_file}")

import os

import pandas as pd
import torch

from data_engine.util import dir_prepare, store_data_with_no_image
from data_engine.pipeline.dpo_reward_pipeline import answer_sampler, logps_calculator, reward_computer, \
    data_pair_builder
from data_engine.dpo_data_filter import filter
from data_engine.pipeline.pipeline import Pipeline


class DPORewardPipeline(Pipeline):
    @classmethod
    def judge_able_to_process(cls, pipeline_name):
        return pipeline_name.lower() == "dpo_reward"

    @classmethod
    def sample_rollout(cls,
                       instruct_model_name: str,
                       instruct_model_path: str,
                       dataset_path: str,
                       sampled_answer_path: str,
                       sample_k: int,
                       work_dir: str,
                       debug: bool):
        answer_sampler.sample_answer(instruct_model_name, instruct_model_path, dataset_path, sampled_answer_path, sample_k)

    @classmethod
    def reward_calculate(cls,
                         reward_model_name: str,
                         reward_model_path: str,
                         instruct_model_name: str,
                         instruct_model_path: str,
                         sampled_answer_path: str,
                         reward_path: str,
                         work_dir: str,
                         debug: bool):
        reward_logps_output_dir = os.path.join(work_dir, "reward_logps")
        instruct_logps_output_dir = os.path.join(work_dir, "instruct_logps")
        dir_prepare(reward_logps_output_dir)
        dir_prepare(instruct_logps_output_dir)
        logps_calculator.main(
            reward_model_name,
            reward_model_path,
            instruct_model_name,
            instruct_model_path,
            sampled_answer_path,
            reward_logps_output_dir,
            instruct_logps_output_dir)
        if torch.distributed.get_rank() == 0:
            rewards = reward_computer.main(instruct_model_path, reward_logps_output_dir, instruct_logps_output_dir)
            step = 5000
            for idx, start in enumerate(range(0, len(rewards), step)):
                temp_data = rewards[start: min(start + step, len(rewards))]
                df = pd.DataFrame(temp_data)
                df.to_parquet(os.path.join(reward_path, f'RLAIF-V-Dataset-reward_{idx:03}-{len(temp_data)}.parquet'))

    @classmethod
    def pair_build_with_filter(cls,
                               reward_path: str,
                               work_dir: str,
                               sample_k: int,
                               rank: int,
                               distance: int,
                               debug: bool):
        rewards = []
        reward_files = [f for f in os.listdir(reward_path) if f.endswith('.parquet')]

        for reward_file in reward_files:
            reward_file_path = os.path.join(reward_path, reward_file)
            reward_df = pd.read_parquet(reward_file_path)
            rewards.append(reward_df)
        rewards = pd.concat(rewards, ignore_index=True).to_dict(orient='records')
        dpo_pair, sum_output, avg_output = data_pair_builder.main(rewards, sample_k, rank, distance)
        if debug:
            store_data_with_no_image(rewards, os.path.join(work_dir, 'debug', 'dpo_pair.json'))
            store_data_with_no_image(sum_output, os.path.join(work_dir, 'debug', 'sum_output.json'))
            store_data_with_no_image(avg_output, os.path.join(work_dir, 'debug', 'avg_output.json'))
        data = filter.main(dpo_pair)
        if debug:
            store_data_with_no_image(rewards, os.path.join(work_dir, 'debug', 'filtered.json'))

        return data

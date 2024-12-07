# dpo_reward_pipeline.py
import os
from typing import Union

import pandas as pd
import torch

from data_engine.util import dir_prepare, store_data_with_no_image
from data_engine.pipeline.dpo_reward_pipeline import (
    answer_sampler,
    logps_calculator,
    reward_computer,
    data_pair_builder
)
from data_engine.dpo_data_filter import filter
from data_engine.pipeline.pipeline import Pipeline


class DPORewardPipeline(Pipeline):
    @classmethod
    def judge_able_to_process(cls, pipeline_name: str) -> bool:
        return "dpo_reward" in pipeline_name.lower()

    @classmethod
    def sample_rollout(cls, **kwargs) -> None:
        required_params = [
            "instruct_model_name",
            "instruct_model_path",
            "dataset_path",
            "sampled_answer_path",
            "sample_k",
            "work_dir",
            "debug"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for sample_rollout in DPORewardPipeline.")

        answer_sampler.sample_answer(
            kwargs["instruct_model_name"],
            kwargs["instruct_model_path"],
            kwargs["dataset_path"],
            kwargs["sampled_answer_path"],
            kwargs["sample_k"]
        )

    @classmethod
    def reward_calculate(cls, **kwargs) -> None:
        required_params = [
            "reward_model_name",
            "reward_model_path",
            "instruct_model_name",
            "instruct_model_path",
            "sampled_answer_path",
            "reward_path",
            "work_dir",
            "debug"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for reward_calculate in DPORewardPipeline.")

        reward_logps_output_dir = os.path.join(kwargs["work_dir"], "reward_logps")
        instruct_logps_output_dir = os.path.join(kwargs["work_dir"], "instruct_logps")
        dir_prepare(reward_logps_output_dir)
        dir_prepare(instruct_logps_output_dir)
        logps_calculator.main(
            kwargs["reward_model_name"],
            kwargs["reward_model_path"],
            kwargs["instruct_model_name"],
            kwargs["instruct_model_path"],
            kwargs["sampled_answer_path"],
            reward_logps_output_dir,
            instruct_logps_output_dir
        )
        if torch.distributed.get_rank() == 0:
            rewards = reward_computer.main(
                kwargs["instruct_model_path"],
                reward_logps_output_dir,
                instruct_logps_output_dir
            )
            step = 5000
            for idx, start in enumerate(range(0, len(rewards), step)):
                temp_data = rewards[start: min(start + step, len(rewards))]
                df = pd.DataFrame(temp_data)
                df.to_parquet(os.path.join(
                    kwargs["reward_path"],
                    f'RLAIF-V-Dataset-reward_{idx:03}-{len(temp_data)}.parquet'
                ))

    @classmethod
    def pair_build_with_filter(cls, **kwargs) -> Union[list, str]:
        required_params = [
            "sampled_answer_path",
            "reward_path",
            "work_dir",
            "sample_k",
            "rank",
            "distance",
            "debug",
            "strict_follow_rank"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for pair_build_with_filter in DPORewardPipeline.")

        rewards = []
        reward_files = [f for f in os.listdir(kwargs["reward_path"]) if f.endswith('.parquet')]

        for reward_file in reward_files:
            reward_file_path = os.path.join(kwargs["reward_path"], reward_file)
            reward_df = pd.read_parquet(reward_file_path)
            rewards.append(reward_df)
        rewards = pd.concat(rewards, ignore_index=True).to_dict(orient='records')
        dpo_pair, sum_output, avg_output = data_pair_builder.main(
            rewards,
            kwargs["sample_k"],
            kwargs["rank"],
            kwargs["distance"]
        )
        if kwargs["debug"]:
            debug_dir = os.path.join(kwargs["work_dir"], 'debug')
            dir_prepare(debug_dir)
            store_data_with_no_image(rewards, os.path.join(debug_dir, 'dpo_pair.json'))
            store_data_with_no_image(sum_output, os.path.join(debug_dir, 'sum_output.json'))
            store_data_with_no_image(avg_output, os.path.join(debug_dir, 'avg_output.json'))
        data = filter.main(dpo_pair)

        return data

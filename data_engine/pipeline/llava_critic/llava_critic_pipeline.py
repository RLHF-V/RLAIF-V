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

class LLaVACriticPipeline(Pipeline):
    @classmethod
    def judge_able_to_process(cls, pipeline_name: str) -> bool:
        return "llava_critic" in pipeline_name.lower()

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
        sampled_answer_file = [f for f in os.listdir(kwargs["sampled_answer_path"]) if f.endswith('.parquet')]

        answers = []
        for answer_file in sampled_answer_file:
            answer_file_path = os.path.join(kwargs["reward_path"], answer_file)
            answer_df = pd.read_parquet(answer_file_path)
            answers.append(answer_df)
        answers = pd.concat(answers, ignore_index=True).to_dict(orient='records')

    @classmethod
    def pair_build_with_filter(cls, **kwargs) -> Union[list, str]:
        pass
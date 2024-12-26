import os
from typing import Union

import pandas as pd
import torch

from data_engine.util import dir_prepare, store_data_with_no_image
from data_engine.pipeline.dpo_reward_pipeline import answer_sampler
from data_engine.pipeline.llava_critic import reward_calculator
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
                raise ValueError(f"Missing parameter '{param}' for sample_rollout in llava critic pipeline.")

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
            "sampled_answer_path",
            "work_dir",
            "python_path",
            "reward_path",
            "reward_model_path",
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for reward calculate in llava critic pipeline.")

        reward_calculator.calculate_reward(
            kwargs["sampled_answer_path"],
            kwargs["work_dir"],
            kwargs["reward_model_path"],
            kwargs["python_path"]
        )

    @classmethod
    def pair_build_with_filter(cls, **kwargs) -> Union[list, str]:
        pass

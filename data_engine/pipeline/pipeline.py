from typing import Union


class Pipeline:
    @classmethod
    def judge_able_to_process(cls, pipeline_name: str) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def sample_rollout(cls, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def reward_calculate(cls, **kwargs) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def pair_build_with_filter(cls, **kwargs) -> Union[list, str]:
        raise NotImplementedError("Subclasses must implement this method.")

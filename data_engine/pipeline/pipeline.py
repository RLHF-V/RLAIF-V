from typing import Union


class Pipeline:
    @classmethod
    def judge_able_to_process(cls, pipeline_name) -> bool:
        raise NotImplementedError

    @classmethod
    def sample_rollout(cls,
                       instruct_model_name: str,
                       instruct_model_path: str,
                       dataset_path: str,
                       sampled_answer_path: str,
                       sample_k: int,
                       work_dir: str,
                       debug: bool) -> None:
        raise NotImplementedError

    @classmethod
    def reward_calculate(cls,
                         reward_model_name: str,
                         reward_model_path: str,
                         instruct_model_name: str,
                         instruct_model_path: str,
                         sampled_answer_path: str,
                         reward_path: str,
                         work_dir: str,
                         debug: bool) -> None:
        raise NotImplementedError

    @classmethod
    def pair_build_with_filter(cls,
                               sampled_answer_path: str,
                               reward_path: str,
                               work_dir: str,
                               sample_k: int,
                               rank: int,
                               distance: int,
                               debug: bool) -> Union[list, str]:
        raise NotImplementedError

import os
import subprocess

import torch

from data_engine.pipeline.pipeline import Pipeline
from data_engine.util import dir_prepare


def run_bash_script(script_path, *args):
    """Helper function to run bash scripts with arguments."""
    command = ['bash', script_path] + list(args)
    subprocess.run(command, check=True)


def get_jsonl_file(path) -> list:
    jsonl_files = [f for f in os.listdir(path) if f.endswith('.jsonl')]
    return jsonl_files


class DivideAndConquerPipeline(Pipeline):
    @classmethod
    def judge_able_to_process(cls, pipeline_name) -> bool:
        return "divide_and_conquer" in pipeline_name.lower()

    @classmethod
    def sample_rollout(cls,
                       instruct_model_name: str,
                       instruct_model_path: str,
                       dataset_path: str,
                       sampled_answer_path: str,
                       sample_k: int,
                       work_dir: str,
                       debug: bool) -> None:
        if torch.distributed.get_rank() == 0:
            script_path = './script/data_gen/llava15/llava15_diverse_gen.sh'
            run_bash_script(script_path, instruct_model_path, sampled_answer_path, dataset_path,
                            get_jsonl_file(dataset_path)[0], str(0), str(-1),
                            str(torch.cuda.device_count()))

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
        if torch.distributed.get_rank() == 0:
            script_path = './script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh'
            answer_file = os.path.join(sampled_answer_path, os.path.basename(get_jsonl_file(sampled_answer_path)[0])[0])
            run_bash_script(script_path, answer_file, '0', '-1', str(torch.cuda.device_count()),
                            str(torch.cuda.device_count()))
            script_path = './script/data_gen/omnilmm/omnilmm_autocheck.sh'
            check_ques_file = ""
            for file in get_jsonl_file(sampled_answer_path):
                if "llama3-8b_divide.gq.qas.jsonl" in file:
                    check_ques_file = file
                    break
            run_bash_script(script_path, reward_model_path, reward_path, sampled_answer_path, check_ques_file, '0', '-1',
                            str(torch.cuda.device_count()))

    @classmethod
    def pair_build_with_filter(cls,
                               sampled_answer_path: str,
                               reward_path: str,
                               work_dir: str,
                               sample_k: int,
                               rank: int,
                               distance: int,
                               debug: bool) -> str:
        if torch.distributed.get_rank() == 0:
            gq_file = ""
            for file in get_jsonl_file(sampled_answer_path):
                if "llama3-8b_divide.gq.jsonl" in file:
                    gq_file = file
                    break
            feedback_file = get_jsonl_file(reward_path)[0]
            script_path = './script/data_gen/construct_pairs.sh'
            run_bash_script(script_path, os.path.join(reward_path, feedback_file), os.path.join(sampled_answer_path, gq_file), str(2))

            script_path = './utils/get_pairs_filter_shorten.py'
            result_dir = os.path.join(work_dir, "dataset")
            dir_prepare(result_dir)
            subprocess.run([
                'python', script_path,
                '--path', os.path.join(reward_path, feedback_file),
                '--save_path', os.path.join(result_dir, "result.jsonl")
            ], check=True)
            return os.path.join(result_dir, "result.jsonl")

import os
import subprocess
import torch
import sys

from data_engine.pipeline.pipeline import Pipeline
from data_engine.util import dir_prepare


def run_bash_script(script_path, *args):
    """Helper function to run bash scripts with arguments."""
    command = ['bash', script_path] + list(args)
    subprocess.run(command, check=True)


def get_jsonl_file(path: str) -> list:
    jsonl_files = [f for f in os.listdir(path) if f.endswith('.jsonl')]
    return jsonl_files


def get_min_len_file(path: str, name_contains: list[str]) -> str:
    file_dict = {}
    min_len = sys.maxsize
    for file in get_jsonl_file(path):
        record = True
        for name in name_contains:
            if name not in file:
                record = False
                break
        if record:
            file_dict[len(file)] = file
            min_len = min(min_len, len(file))
    return file_dict[min_len]


class DivideAndConquerPipeline(Pipeline):
    @classmethod
    def judge_able_to_process(cls, pipeline_name: str) -> bool:
        return "divide_and_conquer" in pipeline_name.lower()

    @classmethod
    def sample_rollout(cls, **kwargs) -> None:
        required_params = [
            "instruct_model_path",
            "sampled_answer_path",
            "dataset_path",
            "work_dir"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for sample_rollout in DivideAndConquerPipeline.")

        if torch.distributed.get_rank() == 0:
            script_path = './script/data_gen/llava15/llava15_diverse_gen.sh'
            run_bash_script(
                script_path,
                kwargs["instruct_model_path"],
                kwargs["sampled_answer_path"],
                kwargs["dataset_path"],
                get_jsonl_file(kwargs["dataset_path"])[0],
                str(0),
                str(-1),
                str(torch.cuda.device_count())
            )

    @classmethod
    def reward_calculate(cls, **kwargs) -> None:
        required_params = [
            "reward_model_name",
            "reward_model_path",
            "sampled_answer_path",
            "work_dir",
            "python_path"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for reward_calculate in DivideAndConquerPipeline.")

        if torch.distributed.get_rank() == 0:
            # omnilmm, changeq, split
            reward_model_path = kwargs["reward_model_path"].split(',')

            changeq = reward_model_path[1].strip()
            split = reward_model_path[2].strip()
            script_path = './script/data_gen/divide_and_conquer/llama3_8b_divide_and_conquer.sh'
            file_name = get_min_len_file(kwargs["sampled_answer_path"], ['diverse_gen'])
            file_name = os.path.basename(file_name)
            answer_file = os.path.join(kwargs["sampled_answer_path"], file_name[:file_name.rfind('.')])
            run_bash_script(
                script_path,
                answer_file,
                '0',
                '-1',
                str(torch.cuda.device_count()),
                str(torch.cuda.device_count()),
                changeq,
                split
            )

            auto_check_model = reward_model_path[0].strip()
            check_ques_file = get_min_len_file(kwargs["sampled_answer_path"],
                                               ['llama3-8b_divide.gq.qas.jsonl', 'diverse_gen'])
            if 'omni' in auto_check_model.lower() or 'omni' in kwargs["reward_model_name"].lower():
                print("OmniLMM as auto check model")
                script_path = './script/data_gen/omnilmm/omnilmm_autocheck.sh'
                run_bash_script(
                    script_path,
                    auto_check_model,
                    kwargs["reward_path"],
                    kwargs["sampled_answer_path"],
                    check_ques_file,
                    '0',
                    '-1',
                    str(torch.cuda.device_count())
                )
            else:
                print("MiniCPM-llama3-v as auto check model")
                script_path = './script/data_gen/minicpm_llama3_v/minicpm_llama3_v_autocheck.sh'
                run_bash_script(
                    script_path,
                    auto_check_model,
                    kwargs["reward_path"],
                    kwargs["sampled_answer_path"],
                    check_ques_file,
                    '0',
                    '-1',
                    kwargs['python_path'],
                    str(torch.cuda.device_count())
                )

    @classmethod
    def pair_build_with_filter(cls, **kwargs) -> str:
        required_params = [
            "sampled_answer_path",
            "reward_path",
            "work_dir",
            "distance"
        ]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter '{param}' for pair_build_with_filter in DivideAndConquerPipeline.")

        if torch.distributed.get_rank() == 0:
            gq_file = get_min_len_file(kwargs["sampled_answer_path"], ['llama3-8b_divide.gq.jsonl'])
            feedback_file = get_min_len_file(kwargs["reward_path"], [])
            script_path = './script/data_gen/construct_pairs.sh'
            run_bash_script(
                script_path,
                os.path.join(kwargs["reward_path"], feedback_file),
                os.path.join(kwargs["sampled_answer_path"], gq_file),
                str(2)
            )

            script_path = './utils/get_pairs_filter_shorten.py'
            file_path = get_min_len_file(kwargs["reward_path"], ['llama3-8b_divide.gq.qas_pair_diff1_samp2.json'])
            result_dir = os.path.join(kwargs["work_dir"], "dataset")
            dir_prepare(result_dir)
            subprocess.run([
                'python', script_path,
                '--path', os.path.join(kwargs["reward_path"], file_path),
                '--save_path', os.path.join(result_dir, "result.jsonl")
            ], check=True)
            return os.path.join(result_dir, "result.jsonl")

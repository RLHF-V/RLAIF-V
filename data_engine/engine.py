import multiprocessing
import os
import sys
import argparse
import pandas as pd
import torch
import torch.distributed as dist

from data_engine.pipeline.divide_and_conquer.divide_and_conquer_pipeline import DivideAndConquerPipeline
from data_engine.pipeline.dpo_reward_pipeline.dpo_reward_pipeline import DPORewardPipeline
from data_engine.util import *

pipelines = [DPORewardPipeline, DivideAndConquerPipeline]


def run(**kwargs):
    pipline = None
    for pipeline_to_judge in pipelines:
        if pipeline_to_judge.judge_able_to_process(kwargs.get("pipeline_name", "")):
            pipline = pipeline_to_judge
            break
    if pipline is None:
        raise ValueError(f"Unsupported pipeline {kwargs.get('pipeline_name', '')}")

    intermediate_step_dir = os.path.join(kwargs["work_dir"], "intermediate_step")

    dist.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if kwargs.get("debug", False):
        if torch.distributed.get_rank() == 0:
            print(
                "You set debug=True, it will generate fine-grained process data under subdir 'debug'. You can check that dir for debug details."
            )

    # 0: sample answer
    sampled_answer_path = os.path.join(kwargs["work_dir"], "sampled_answer")
    if kwargs.get("run_stage", 0) == 0:
        dir_prepare(sampled_answer_path)
        sub_work_dir = os.path.join(intermediate_step_dir, "sample_answers")
        dir_prepare(sub_work_dir)
        print_stage(0, "Sample answers")
        pipline.sample_rollout(
            instruct_model_name=kwargs["instruct_model_name"],
            instruct_model_path=kwargs["instruct_model_path"],
            dataset_path=kwargs["dataset_path"],
            sampled_answer_path=sampled_answer_path,
            sample_k=kwargs["sample_k"],
            work_dir=sub_work_dir,
            debug=kwargs["debug"],
            # 对于 DPORewardPipeline，可能需要额外参数，如 strict_follow_rank
            strict_follow_rank=kwargs.get("strict_follow_rank", False)
        )
        print_stage(0, finish=True)

    # 1: calculate rewards
    reward_output_dir = os.path.join(kwargs["work_dir"], "reward")
    if kwargs.get("run_stage", 0) == 1:
        dir_prepare(reward_output_dir)
        sub_work_dir = os.path.join(intermediate_step_dir, "calculate_rewards")
        dir_prepare(sub_work_dir)
        print_stage(1, "Calculate rewards")
        pipline.reward_calculate(
            reward_model_name=kwargs["reward_model_name"],
            reward_model_path=kwargs["reward_model_path"],
            instruct_model_name=kwargs["instruct_model_name"],
            instruct_model_path=kwargs["instruct_model_path"],
            sampled_answer_path=sampled_answer_path,
            reward_path=reward_output_dir,
            work_dir=sub_work_dir,
            python_path=kwargs["reward_model_python_path"],
            debug=kwargs["debug"],
        )
        print_stage(1, finish=True)

    # following code doesn't need multi CUDA
    if torch.distributed.get_rank() == 0:
        if kwargs.get("run_stage", 0) == 2:
            print_stage(2, "Pair build and filter")
            sub_work_dir = os.path.join(intermediate_step_dir, "pair_build_and_filter")
            dir_prepare(sub_work_dir)
            pair_build_kwargs = {"sampled_answer_path": sampled_answer_path, "reward_path": reward_output_dir,
                                 "work_dir": sub_work_dir, "sample_k": kwargs["sample_k"], "rank": kwargs["rank"],
                                 "distance": kwargs["distance"], "debug": kwargs["debug"],
                                 "strict_follow_rank": kwargs.get("strict_follow_rank", False)}

            data = pipline.pair_build_with_filter(**pair_build_kwargs)
            print_stage(2, finish=True)

            if isinstance(data, str):
                print(f"Dataset stored to {data}")
            else:
                # -1: save files
                print_stage(-1, "Save file to dataset format")
                output_path = os.path.join(kwargs["work_dir"], "dataset")
                output_file = os.path.join(output_path, "dpo_dataset.parquet")
                dir_prepare(output_path)
                needed_keys = [
                    "question",
                    "chosen",
                    "rejected",
                    "origin_dataset",
                    "origin_split",
                    "idx",
                    "image_path",
                    "ds_name",
                    "image"
                ]
                for item in data:
                    for key in list(item.keys()):
                        if key not in needed_keys:
                            del item[key]
                df = pd.DataFrame(data)
                df = df.sample(frac=1).reset_index(drop=True)
                df.to_parquet(output_file)
                print_stage(-1)

                print(f"We get {len(data)} data items in total, you may need that to set max_steps for training")
                print("Finish all stages, output file is saved to ", output_path)
                print("You can directly copy this path to the training script to replace --data_dir value")
                print("Have a nice day!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Data Pipeline")
    parser.add_argument("--reward_model_name", type=str, required=True, help="The name of the reward model.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="The path of the reward model.")
    parser.add_argument("--instruct_model_name", type=str, required=True, help="The name of the instruct model.")
    parser.add_argument("--instruct_model_path", type=str, required=True, help="The path of the instruct model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="The path of the dataset.")
    parser.add_argument("--work_dir", type=str, required=True, help="The working directory.")
    parser.add_argument("--pipeline_name", type=str, required=True, help="The pipeline you choose to run.")
    parser.add_argument("--run_stage", type=int, default=0, help="The stage to run.")
    parser.add_argument("--sample_k", type=int, default=10, help="The sample number k.")
    parser.add_argument("--rank", type=int, default=3, help="The rank number. (specific to DPORewardPipeline)")
    parser.add_argument("--distance", type=int, default=25, help="The distance. (specific to DPORewardPipeline)")
    parser.add_argument('--reward_model_python_path', type=str, help="Python path to reward model. Not required for all pipelines.")
    parser.add_argument("--debug", action='store_true', help="Preserve fine-grained process data")
    parser.add_argument("--strict_follow_rank", action='store_true',
                        help="Strictly follow rank (specific to DPORewardPipeline)")

    args = parser.parse_args()

    run(
        reward_model_name=args.reward_model_name,
        reward_model_path=args.reward_model_path,
        instruct_model_name=args.instruct_model_name,
        instruct_model_path=args.instruct_model_path,
        dataset_path=args.dataset_path,
        work_dir=args.work_dir,
        pipeline_name=args.pipeline_name,
        run_stage=args.run_stage,
        sample_k=args.sample_k,
        rank=args.rank,
        distance=args.distance,
        debug=args.debug,
        strict_follow_rank=args.strict_follow_rank,
        reward_model_python_path=args.reward_model_python_path,
    )

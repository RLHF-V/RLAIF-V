import os.path
import random

import pandas as pd

from data_engine.pipeline.dpo_reward_pipeline.dpo_reward_pipeline import DPORewardPipeline
from data_engine.util import *
import argparse
import torch
import torch.distributed as dist

pipelines = [DPORewardPipeline]


def run(
        reward_model_name,
        reward_model_path,
        instruct_model_name,
        instruct_model_path,
        dataset_path,
        work_dir,
        pipeline_name,
        continue_from_stage=1,
        sample_k=10,
        rank=3,
        distance=25,
        debug=False
):
    pipline = None
    for pipeline_to_judge in pipelines:
        if pipeline_to_judge.judge_able_to_process(pipeline_name):
            pipline = pipeline_to_judge
            break
    if pipline is None:
        raise ValueError("Unsupported pipeline")

    intermediate_step_dir = os.path.join(work_dir, "intermediate_step")
    if debug:
        print(
            "You set debug=True, it will generate fine-grained process data under subdir 'debug'. You can check that dir for debug details.")

    dist.init_process_group(backend='nccl', world_size=int(os.getenv('WORLD_SIZE', '1')),
                            rank=int(os.getenv('RANK', '0')), )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # 0: sample answer
    sampled_answer_path = os.path.join(work_dir, "sampled_answer")
    if continue_from_stage <= 0:
        print_stage(0, "Sample answers")
        pipline.sample_rollout(
            instruct_model_name,
            instruct_model_path,
            dataset_path,
            sampled_answer_path,
            sample_k,
            os.path.join(intermediate_step_dir, "sample_answers"),
            debug
        )
        print_stage(0, finish=True)

    # 1: calculate logps
    reward_output_dir = os.path.join(work_dir, "reward")
    if continue_from_stage <= 1:
        print_stage(1, "Calculate rewards")
        pipline.reward_calculate(
            reward_model_name,
            reward_model_path,
            instruct_model_name,
            instruct_model_path,
            sampled_answer_path,
            reward_output_dir,
            os.path.join(intermediate_step_dir, "calculate_rewards"),
            debug
        )
        print_stage(1, finish=True)

    # following code doesn't need multi CUDA
    if torch.distributed.get_rank() == 0:
        if continue_from_stage <= 2:
            print_stage(2, "Pair build and filter")

            data = pipline.pair_build_with_filter(
                reward_output_dir,
                os.path.join(intermediate_step_dir, "pair_build_and_filter"),
                sample_k,
                rank,
                distance,
                debug
            )
            print_stage(2, finish=True)

            # -1: save files
            print_stage(-1, "Save file to dataset format")
            output_path = os.path.join(work_dir, "dataset")
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
                "image"]
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
    args = argparse.ArgumentParser()
    args.add_argument("--reward_model_name", type=str, help="The name of the reward model.")
    args.add_argument("--reward_model_path", type=str, help="The path of the reward model.")
    args.add_argument("--instruct_model_name", type=str, help="The name of the instruct model.")
    args.add_argument("--instruct_model_path", type=str, help="The path of the instruct model.")
    args.add_argument("--dataset_path", type=str, help="The path of the dataset.")
    args.add_argument("--work_dir", type=str, help="The working directory.")
    args.add_argument("--continue_from_stage", type=int, default=1, help="The stage to continue from.")
    args.add_argument("--sample_k", type=int, default=10, help="The sample number k.")
    args.add_argument("--rank", type=int, default=3, help="The rank number.")
    args.add_argument("--distance", type=int, default=25, help="The distance.")
    args.add_argument("--debug", type=bool, default=False, help="Preserve fine-grained process data")

    args = args.parse_args()
    run(
        args.reward_model_name,
        args.reward_model_path,
        args.instruct_model_name,
        args.instruct_model_path,
        args.dataset_path,
        args.work_dir,
        args.continue_from_stage,
        args.sample_k,
        args.rank,
        args.distance,
        args.debug
    )

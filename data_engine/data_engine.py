import os.path

import pandas as pd

import logps_calculator
import reward_computer
import data_pair_builder
from dpo_data_filter import filter
import answer_sampler
import argparse
import torch
import torch.distributed as dist


def print_stage(idx, desc="", finish=False):
    if torch.distributed.get_rank() == 0:
        print("=" * 80)
        if not finish:
            print(f"Processing Stage {idx}: {desc}")
        else:
            print(f"Finish Stage {idx}")
        print("=" * 80)


def dir_prepare(dir_to_check, clean=True):
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(dir_to_check):
            os.makedirs(dir_to_check)
        elif clean:
            if os.path.isdir(dir_to_check):
                for file in os.listdir(dir_to_check):
                    os.remove(os.path.join(dir_to_check, file))
            else:
                os.remove(dir_to_check)
                os.mkdir(dir_to_check)


def run(
        reward_model_name,
        reward_model_path,
        instruct_model_name,
        instruct_model_path,
        dataset_path,
        work_dir,
        image_column="image",
        continue_from_stage=1,
        sample_k=10,
        rank=10,
        distance=5
):
    dist.init_process_group(backend='nccl', world_size=int(os.getenv('WORLD_SIZE', '1')),
                            rank=int(os.getenv('RANK', '0')), )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    sampled_answer_path = os.path.join(work_dir, "sampled_answer")
    if continue_from_stage <= 0:
        print_stage(0, "Sample answers")
        dir_prepare(sampled_answer_path)
        answer_sampler.sample_answer(instruct_model_path, dataset_path, sampled_answer_path, image_column, sample_k)
        print_stage(0, finish=True)

    reward_logps_output_dir = os.path.join(work_dir, "reward_logps")
    instruct_logps_output_dir = os.path.join(work_dir, "instruct_logps")
    if continue_from_stage <= 1:
        print_stage(1, "Calculate logps")
        dir_prepare(reward_logps_output_dir)
        dir_prepare(instruct_logps_output_dir)
        _ = logps_calculator.main(
            reward_model_name,
            reward_model_path,
            instruct_model_name,
            instruct_model_path,
            sampled_answer_path,
            reward_logps_output_dir,
            instruct_logps_output_dir)
        print_stage(1, finish=True)

    # following code doesn't need multi CUDA
    if torch.distributed.get_rank() == 0:
        if continue_from_stage <= 2:
            print_stage(2, "DPO dataset construction")

            print_stage(2.1, "Calculate reward")
            rewards = reward_computer.main(instruct_model_path, reward_logps_output_dir, instruct_logps_output_dir)
            print_stage(2.1, finish=True)

            print_stage(2.2, "Build DPO pairs")
            dpo_pair = data_pair_builder.main(rewards, sample_k, rank, distance)
            print_stage(2.2, finish=True)

            print_stage(2.3, "Filter DPO pairs")
            data = filter.main(dpo_pair)
            print_stage(2.3, finish=True)

            print_stage(2.4, "Save file to dataset format")
            output_file = os.path.join(work_dir, "dataset", "dpo_dataset.parquet")
            if os.path.exists(output_file):
                os.remove(output_file)
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
            df.to_parquet(output_file)
            print_stage(2.4, finish=True)

            print_stage(2, finish=True)

            print("Finish all stages, output file is saved to ", output_file)
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
    args.add_argument("--image_column", type=str, help="The column that keep image in your dataset")
    args.add_argument("--continue_from_stage", type=int, default=1, help="The stage to continue from.")
    args.add_argument("--sample_k", type=int, default=10, help="The sample number k.")
    args.add_argument("--rank", type=int, default=10, help="The rank number.")
    args.add_argument("--distance", type=int, default=5, help="The distance.")

    args = args.parse_args()
    run(
        args.reward_model_name,
        args.reward_model_path,
        args.instruct_model_name,
        args.instruct_model_path,
        args.dataset_path,
        args.work_dir,
        args.image_column,
        args.continue_from_stage,
        args.sample_k,
        args.rank,
        args.distance
    )

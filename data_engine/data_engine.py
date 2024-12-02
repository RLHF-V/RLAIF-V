import json
import os.path
import random
from copy import deepcopy

import pandas as pd

import logps_calculator
import reward_computer
import data_pair_builder
from dpo_data_filter import filter
import answer_sampler
import argparse
import torch
import torch.distributed as dist


def store_data_with_no_image(data, path):
    if torch.distributed.get_rank() == 0:
        data_to_store = []
        for item in data:
            item = deepcopy(item)
            item.pop('image', None)
            data_to_store.append(item)

        with open(path, 'w') as f:
            json.dump(data_to_store, f, ensure_ascii=False, indent=4)


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
        continue_from_stage=1,
        sample_k=10,
        rank=3,
        distance=25,
        debug=False
):
    # -1: multi cuda env init
    dist.init_process_group(backend='nccl', world_size=int(os.getenv('WORLD_SIZE', '1')),
                            rank=int(os.getenv('RANK', '0')), )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # 0: sample answer
    sampled_answer_path = os.path.join(work_dir, "sampled_answer")
    if continue_from_stage <= 0:
        print_stage(0, "Sample answers")
        dir_prepare(sampled_answer_path)
        answer_sampler.sample_answer(instruct_model_name, instruct_model_path, dataset_path, sampled_answer_path,
                                     sample_k)
        print_stage(0, finish=True)

    # 1: calculate logps
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
        debug_root_dir = os.path.join(work_dir, 'debug')
        if debug:
            print(
                "You set debug=True, it will generate fine-grained process data under subdir 'debug'. You can check that dir for debug details.")
            dir_prepare(debug_root_dir)
        if continue_from_stage <= 2:
            print_stage(2, "DPO dataset construction")

            # 2.1: calculate reward
            print_stage(2.1, "Calculate reward")
            rewards = reward_computer.main(instruct_model_path, reward_logps_output_dir, instruct_logps_output_dir)
            if debug:
                store_data_with_no_image(rewards, os.path.join(debug_root_dir, 'rewards.json'))
            print_stage(2.1, finish=True)

            # 2.2: build DPO pair
            print_stage(2.2, "Build DPO pairs")
            dpo_pair, sum_output, avg_output = data_pair_builder.main(rewards, sample_k, rank, distance)
            if debug:
                store_data_with_no_image(rewards, os.path.join(debug_root_dir, 'dpo_pair.json'))
                store_data_with_no_image(sum_output, os.path.join(debug_root_dir, 'sum_output.json'))
                store_data_with_no_image(avg_output, os.path.join(debug_root_dir, 'avg_output.json'))
            print_stage(2.2, finish=True)

            # 2.3: filter DPO pairs
            print_stage(2.3, "Filter DPO pairs")
            data = filter.main(dpo_pair)
            if debug:
                store_data_with_no_image(rewards, os.path.join(debug_root_dir, 'filtered.json'))
            print_stage(2.3, finish=True)

            # 2.4: save files
            print_stage(2.4, "Save file to dataset format")
            output_path = os.path.join(work_dir, "dataset")
            output_file = os.path.join(output_path, "dpo_dataset.parquet")
            random.shuffle(data)
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
            print_stage(2.4, finish=True)

            print_stage(2, finish=True)

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

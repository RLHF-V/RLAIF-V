import jsonlines
import pandas as pd
import pyarrow.parquet as pq
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse


# def parquet_to_json(parquet_file, jsonl_file):
#     df = pd.read_parquet(parquet_file, engine='pyarrow')
#
#     df = df.astype(str, errors='ignore')
#     df.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)
#
# def compute_reward(tokenizer, reward_logps_dir, instruct_logps_dir):
#     rewards = []
#     reward_files = [f for f in os.listdir(reward_logps_dir) if f.endswith('.parquet')]
#
#     for reward_file in tqdm(reward_files, desc='Processing files'):
#         suffix = reward_file.split('_')[-1].split('.')[0]
#         reward_file_path = os.path.join(reward_logps_dir, reward_file)
#         instruct_file_path = os.path.join(instruct_logps_dir, f'RLAIF-V-Dataset-withlogp_{suffix}.parquet')
#
#         reward_jsonl_file = os.path.join(reward_logps_dir, reward_file.replace('.parquet', '.jsonl'))
#         instruct_jsonl_file = os.path.join(instruct_logps_dir, instruct_file_path.replace('.parquet', '.jsonl'))
#
#         if not os.path.exists(reward_jsonl_file):
#             print(f"Converting {reward_file_path} to {reward_jsonl_file}. For each file, it will only perform once, please wait...")
#             # convert_parquet_to_jsonl(reward_file_path, reward_jsonl_file)
#             parquet_to_json(reward_file_path, reward_jsonl_file)
#             print(f'Successfully converted {reward_file_path} to {reward_jsonl_file}')
#         if not os.path.exists(instruct_jsonl_file):
#             print(f"Converting {instruct_file_path} to {instruct_jsonl_file}. F or each file, it will only perform once, please wait...")
#             # convert_parquet_to_jsonl(instruct_file_path, instruct_jsonl_file)
#             parquet_to_json(instruct_file_path, instruct_jsonl_file)
#             print(f'Successfully converted {instruct_file_path} to {instruct_jsonl_file}')
#
#         with jsonlines.open(reward_jsonl_file) as reward_reader, jsonlines.open(
#                 instruct_jsonl_file) as instruct_reader:
#             for obj in reward_reader:
#                 idx = obj["idx"]
#                 tokens = tokenizer.encode(obj["chosen"])
#                 logps = obj["logps"].split("[")[-1].split("]")[0]
#                 reward_logps = list(map(float, logps.split(",")))
#                 reward_logps_for_reward = reward_logps[-len(tokens):]
#
#                 for instruct_obj in instruct_reader:
#                     if instruct_obj["idx"] == idx:
#                         instruct_logps = instruct_obj["logps"].split("[")[-1].split("]")[0]
#                         instruct_logps = list(map(float, instruct_logps.split(",")))
#                         instruct_logps_for_reward = instruct_logps[-len(tokens):]
#                         break
#
#                 differences = [instruct_logp - reward_logp for instruct_logp, reward_logp in
#                                zip(instruct_logps_for_reward, reward_logps_for_reward)]
#                 min_reward = min(differences) * 0.1
#                 sum_reward = sum(differences) * 0.1
#                 last_reward = differences[-1] * 0.1
#                 avg_reward = sum_reward / len(tokens) * 0.1
#
#                 reward_data = {
#                     "idx": idx,
#                     "ds_name": obj["ds_name"],
#                     "question": obj["question"],
#                     "chosen": obj["chosen"],
#                     "image": obj["image"],
#                     "image_path": obj["image_path"],
#                     "origin_split": obj["origin_split"],
#                     "origin_dataset": obj["origin_dataset"],
#                     "min": min_reward,
#                     "sum": sum_reward,
#                     "ORM": last_reward,
#                     "avg": avg_reward
#                 }
#
#                 rewards.append(reward_data)
#     return rewards

def compute_reward(tokenizer, reward_logps_dir, instruct_logps_dir):
    rewards = []
    reward_files = [f for f in os.listdir(reward_logps_dir) if f.endswith('.parquet')]

    for reward_file in tqdm(reward_files, desc='Processing files'):
        suffix = reward_file.split('_')[-1].split('.')[0]
        reward_file_path = os.path.join(reward_logps_dir, reward_file)
        instruct_file_path = os.path.join(instruct_logps_dir, f'RLAIF-V-Dataset-withlogp_{suffix}.parquet')

        reward_df = pd.read_parquet(reward_file_path)
        instruct_df = pd.read_parquet(instruct_file_path)

        for _, reward_row in reward_df.iterrows():
            idx = reward_row["idx"]
            tokens = tokenizer.encode(reward_row["chosen"])
            logps = reward_row["logps"].split("[")[-1].split("]")[0]
            reward_logps = list(map(float, logps.split(",")))
            reward_logps_for_reward = reward_logps[-len(tokens):]

            instruct_row = instruct_df[instruct_df["idx"] == idx].iloc[0]
            instruct_logps = instruct_row["logps"].split("[")[-1].split("]")[0]
            instruct_logps = list(map(float, instruct_logps.split(",")))
            instruct_logps_for_reward = instruct_logps[-len(tokens):]

            differences = [instruct_logp - reward_logp for instruct_logp, reward_logp in
                           zip(instruct_logps_for_reward, reward_logps_for_reward)]
            min_reward = min(differences) * 0.1
            sum_reward = sum(differences) * 0.1
            last_reward = differences[-1] * 0.1
            avg_reward = sum_reward / len(tokens) * 0.1

            reward_data = {
                "idx": idx,
                "ds_name": reward_row["ds_name"],
                "question": reward_row["question"],
                "chosen": reward_row["chosen"],
                "image": reward_row["image"],
                "image_path": reward_row["image_path"],
                "origin_split": reward_row["origin_split"],
                "origin_dataset": reward_row["origin_dataset"],
                "min": min_reward,
                "sum": sum_reward,
                "ORM": last_reward,
                "avg": avg_reward
            }

            rewards.append(reward_data)

    return rewards


def main(model_path: str, reward_logps_dir: str, instruct_logps_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rewards = compute_reward(tokenizer, reward_logps_dir, instruct_logps_dir)
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="turn logps to reward")
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--reward_logps_dir", type=str, help="reward logps dir")
    parser.add_argument("--instruct_logps_dir", type=str, help="instruct logps dir")
    parser.add_argument("--output_file", type=str, help="output file")
    args = parser.parse_args()
    rewards = main(args.model_path, args.reward_logps_dir, args.instruct_logps_dir)
    with jsonlines.open(args.output_file, 'w') as writer:
        writer.write_all(rewards)

import json
import os
import shutil
from copy import deepcopy
import glob

import pandas as pd
import torch


def judge_is_llava(model_name: str) -> bool:
    lower_name = model_name.lower()
    return 'llava' in lower_name or ('rlaif' in lower_name and '7b' in lower_name)


def judge_is_omnilmm(model_name: str) -> bool:
    lower_name = model_name.lower()
    return 'omnilmm' in lower_name or ('rlaif' in lower_name and '12b' in lower_name)


def judge_is_minicpmv26(model_name: str) -> bool:
    lower_name = model_name.lower()
    return ("minicpm-v" in lower_name or "minicpm_v" in lower_name) and (
            "2_6" in lower_name or "2.6" in lower_name or "26" in lower_name)


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
                shutil.rmtree(dir_to_check)
            else:
                os.remove(dir_to_check)
            os.makedirs(dir_to_check)


def read_parquets(dir_path):
    parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))
    if not parquet_files:
        raise ValueError(f"No .parquet files found in directory: {dir_path}")

    # Initialize an empty list to collect all data
    all_data = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            data = df.to_dict(orient='records')
            all_data.extend(data)
            print(f"Loaded {len(data)} records from {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        raise ValueError("No data loaded from the .parquet files.")

    return all_data

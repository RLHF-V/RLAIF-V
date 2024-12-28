import pandas as pd
from tqdm import tqdm

from data_engine.util import read_parquets
from data_engine.dpo_data_filter import filter


def build_and_filter(input_dir: str, rank: int):
    """
    Processes the output .parquet files by grouping data by 'idx', sorting by 'score',
    and selecting top and next 'rank' entries as 'chosen' and 'rejected' respectively.
    The final data includes 'idx', 'question', 'chosen', 'rejected', and 'image' fields.

    Args:
        input_dir (str): Path to the directory containing the input .parquet files.
        rank (int): Number of top and next entries to select for 'chosen' and 'rejected'.
    """

    # Read all data from input .parquet files
    all_data = read_parquets(input_dir)

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(all_data)

    # Check required columns
    required_columns = {'idx', 'score', 'chosen', 'question', 'image'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise KeyError(f"Missing required columns in input data: {missing}")

    # Group by 'idx'
    grouped = df.groupby('idx')

    processed_records = []

    print("Processing groups...")
    for idx, group in tqdm(grouped, desc="Processing groups"):
        # Sort the group by 'score' descending
        sorted_group = group.sort_values(by='score', ascending=False).reset_index(drop=True)

        # Select top 'rank' as 'chosen'
        chosen_subset = sorted_group.head(rank)

        # Select next 'rank' as 'rejected'
        rejected_subset = sorted_group.iloc[rank:rank * 2]

        # Ensure that there are enough entries
        if len(rejected_subset) < rank:
            print(f"Warning: Not enough rejected entries for idx {idx}. Expected {rank}, got {len(rejected_subset)}.")

        # Pair 'chosen' and 'rejected'
        for i in range(min(len(chosen_subset), len(rejected_subset))):
            chosen_entry = chosen_subset.iloc[i]
            rejected_entry = rejected_subset.iloc[i]

            record = {
                'idx': idx,
                'question': chosen_entry['question'],  # Assuming question is the same within the group
                'chosen': chosen_entry['chosen'],
                'rejected': rejected_entry['chosen'],
                'image': chosen_entry['image']  # Assuming image is the same within the group
            }
            processed_records.append(record)

    print(f"Total processed records: {len(processed_records)}")
    processed_records = filter.main(processed_records)

    return processed_records

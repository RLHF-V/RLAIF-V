import json
import os.path

import jsonlines


class Filter:
    @classmethod
    def do_filter(cls, data: list) -> list:
        """

        Args:
            data: (list): data need to be filtered

        Returns: (list): filtered data
        """
        raise NotImplementedError


def jsonl_to_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
    return data


def json_to_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def load_data(file_path):
    _, ext = os.path.splitext(file_path)

    if ext == '.jsonl':
        return jsonl_to_data(file_path)
    elif ext == '.json':
        return json_to_data(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def main(data):
    # import filters here to avoid circulate important
    from .length_filter import LengthFilter
    from .num_filter import NumFilter
    from .same_filter import DeleteSameFilter

    # you can add your own filters here or delete the filters
    # that are determined to be unnecessary
    filters = [LengthFilter, NumFilter, DeleteSameFilter]

    for filter_to_run in filters:
        filter_name = filter_to_run.__class__.__name__
        filter_doc = filter_to_run.__doc__ if filter_to_run.__doc__ else "No documentation available"
        print("=" * 80)
        print(f"Processing Filter: {filter_name}")
        print("=" * 80)
        print(f"Documentation:\n{filter_doc}\n")

        data = filter_to_run.do_filter(data)

        print("=" * 80)
        print(f"Filter {filter_name} Finished")
        print(f"After filtering, we get {len(data)} data items")
        print("=" * 80 + "\n")

    return data
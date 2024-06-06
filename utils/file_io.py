import json
import jsonlines

def write_jsonlines(file: str, dataset: list):
    with jsonlines.open(file, 'w') as writer:
        for data in dataset:
            writer.write(data)

def write_json(save_path, data):
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data

def read_jsonlines(file: str, index_begin = None, index_all = None):
    dataset = []
    if index_begin == None:
        index_begin = 0
        index_all = -1
    with jsonlines.open(file, 'r') as reader:
        for data in reader:
            if index_begin != 0:
                index_begin -= 1
                continue
            dataset += [data]
            index_all -= 1
            if index_all == 0:
                break
    return dataset
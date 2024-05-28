import os
import sys
import glob
import numpy as np
import json
import jsonlines

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data


def cal_informative(path, return_meta=False):
    responses = read_json(path)

    scores = []
    for i, response in enumerate(responses):
        response = response['choices'][0]['message']['content']
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    informativeness = []
    for s in scores:
        if s >= 3:
            informativeness.append(s-3)
        else:
            informativeness.append(s)

    mean_informativeness = np.mean(informativeness)/3 * 100
    print("Informativeness: {:.2f}".format(mean_informativeness))

    if return_meta:
        return informativeness, mean_informativeness
    else:
        return mean_informativeness

def cal_mmhalscore(path):
    responses = read_json(path)

    scores = []
    for i, response in enumerate(responses):
        response = response['choices'][0]['message']['content']
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join([str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]), flush=True)

if __name__ == '__main__':
    base_dir = sys.argv[1]
    print(base_dir)

    patterns = ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*']
    f_list = sum([list(glob.glob(base_dir + p)) for p in patterns], [])
    review_files = [x for x in f_list if x.endswith('.json') and '.mmhal_test_eval' in x]

    for file in review_files:
        print("===>", file)
        informativeness = cal_informative(file)
        cal_mmhalscore(file)
import re
import os
import time
import tqdm
import glob
import json
import pathlib
import argparse
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import random
from datetime import datetime

from gpt4 import Chat, get_eval

SYSTEM_MSG = '''
There is currently a review opinion comparing two responses that urgently need to be analyzed. The review is conducted based on two criteria including the hallucination rate and helpfulness.

We greatly need you to act as an impartial expert and provide valuable analyzing results. Based on the review text, you need to carefully summarize which model has fewer hallucinations.

When you output your evaluation opinions to users, we hope you strictly follow the following format:

Firstly, please evaluate the number of hallucinations of model A and model B.

Secondly, please strictly output your final conclusion in the following format:

* If both models have the same number of hallucinations or both have no hallucinations, output \"[[C]]\";
* If model A has fewer hallucinations, output  \"[[A]]\";
* If model B has fewer hallucinations output \"[[B]]\";

'''

def construct_gpt4_query(review_text):
    prompt = f'''
    {SYSTEM_MSG}

    [Beginning of the review text]
    {review_text}
    [End of the review text]

    '''
    return prompt


def post_process(output):
    match = re.findall(r'\[\[(A|B|C)\]\]', output)[0]

    if 'A' in match:
        score = -1
    elif 'B' in match:
        score = 1
    else:
        score = 0

    review = output
    return score, review


def read_jsonl_modelA_0504(file_path):
    """用于读取 JSONL 文件，并返回一个以 (image_id, Question) 作为键的字典"""
    data_dict = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        data_dict.append(
            {
                'review' : item['review'],
                'type' : item['type_name'],
                'model_B' : item['modelB'],
            })
    return data_dict

def check_keys(data_list):
    for index, item in enumerate(data_list):
        if 'review' not in item:
            raise Exception(f"Key 'modelB answer' is missing in the dictionary at index {index}")
        else:
            # print(f"Record {index} is complete. Contains 'modelB answer'.")
            pass

#  python eval_0522_hallucination.py --jsonl_file
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OpenMME evaluation')
    parser.add_argument('--jsonl_file', type=str,
                        default=None)

    args = parser.parse_args()

    ref_model_list = ['GPT-4V']

    file1_data_modelA = read_jsonl_modelA_0504(args.jsonl_file)

    model = "gpt-4-1106-preview"
    chat = Chat(model=model, timeout_sec=120)

    try:
        check_keys(file1_data_modelA)
    except Exception as e:
        print(e)


    # 生成新的列表
    reviews =[]
    type_name_list = []
    model_B_list =[]


    # 将字典中的每个值添加到新列表中
    for item in file1_data_modelA:
        reviews.append(item['review'])
        type_name_list.append(item['type'])
        model_B_list.append(item['model_B'])

    print(f'Evaluating {args.jsonl_file}')

    reviews_out_put =[]
    with ThreadPoolExecutor(max_workers=128) as executor:
        tasks = []
        modelname = []
        imageid = []
        token = 0
        in_token = 0
        out_token = 0
        def eval_worker(x, review_text, type_name, model_B):
            while True:
                response, org_resp = get_eval(chat, x, chat_gpt_system="You are a helpful and precise assistant.",
                                              max_tokens=2048, top_p=1.0, temperature=0.0) ###
                try:
                    score, review_res = post_process(response)
                    out = {
                        'score': score,
                        'review_text': review_text,
                        'review_res': review_res,
                        'type_name': type_name,
                        'modelA': 'GPT-4V',
                        'modelB': model_B
                    }
                    return out
                except:
                    print(f'Fail parsing {response}')
                    continue


        for qid, review in enumerate(reviews):

            review_text = review
            type_name = type_name_list[qid]
            model_B = model_B_list[qid]

            prompt = construct_gpt4_query(review_text = review_text)

            tasks.append(executor.submit(eval_worker, str(prompt), review_text,type_name, model_B  ))


        pb = tqdm.tqdm(total=len(tasks))

        for i, future in enumerate(concurrent.futures.as_completed(tasks)):
            pb.update(1)
            try:
                new_data_item = future.result()
                reviews_out_put.append(new_data_item)
                json.dump(reviews_out_put, open(args.jsonl_file.replace('.json', '.hall.temp'), 'w'),
                        indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"@@@ Exception: {e}\n")

        sum_score = 0
        sum_cnt = 0
        for item in reviews_out_put:
            sum_score += (item['score'] + 1)/2.0
            sum_cnt += 1
        print(f'Score is {sum_score / sum_cnt:.3f}')

        json.dump(reviews_out_put, open(args.jsonl_file.replace('.json', '.hall.json'), 'w'),
                    indent=2, ensure_ascii=False)

        os.system(f"rm {args.jsonl_file.replace('.json', '.hall.temp')}")
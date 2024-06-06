import random
import argparse
from collections import defaultdict

from pair_construction import get_pairs_inner
from file_io import read_jsonlines, read_json, write_jsonlines


def filter_same_instruct(org_data, autocheck_data):
    used_imgpath = [f"{org_data[0]['ds_question_id']}@{org_data[0]['raw_question']}"]
    new_data = []
    curr_img = f"{org_data[0]['ds_question_id']}@{org_data[0]['raw_question']}"
    for item in org_data:
        temp_key = f"{item['ds_question_id']}@{item['raw_question']}"
        if temp_key == curr_img:
            new_data.append(item)
        else:
            if temp_key not in used_imgpath:
                used_imgpath.append(temp_key)
                curr_img = temp_key
                new_data.append(item)
            else:
                print(temp_key)
                continue

    print("diff instruction num:", len(used_imgpath))

    ques_ids = set([item['question_id'] for item in new_data])
    remain_autocheck = []
    for item in autocheck_data:
        if item['question_id'] in ques_ids:
            remain_autocheck.append(item)

    return new_data, remain_autocheck

def save_pred_quesid_to_judge(pred_quesid_to_judge, origin_divide_data, save_path):
    new_data = []
    for item in origin_divide_data:
        item['fact_judge'] = pred_quesid_to_judge[item['question_id']]
        new_data.append(item)

    write_jsonlines(save_path, new_data)


def get_pair_data(path_autocheck, path_ans_divide, save_path, diff=1):
    try:
        data = read_json(path_autocheck)
    except:
        data = read_jsonlines(path_autocheck)
    print("origin facts len:", len(data))


    try:
        origin_divide_data = read_jsonlines(path_ans_divide)
    except:
        origin_divide_data = read_json(path_ans_divide)

    print("origin data len:", len(origin_divide_data))

    origin_divide_data, data = filter_same_instruct(origin_divide_data, data)
    print("filtered_origin:", len(origin_divide_data), "filtered_autocheck:", len(data))


    question_id_2_origin_data = {item['question_id']: item for item in origin_divide_data}
    assert len(question_id_2_origin_data) == len(origin_divide_data), f'{len(question_id_2_origin_data)},{len(origin_divide_data)}'

    ### pair data format
    # chosen_pair_data = {
    #     "ds_question_id": key,
    #     "chosen": {"question_id": comp_idx1, "score": ans_1_score},
    #     "rejected": {"question_id": comp_idx2, "score": ans_2_score},
    # }
    pair_data_ids, pred_quesid_to_judge, pred_addcls = get_pairs_inner(data, diff=diff, return_infos=True)

    pair_data = []
    ch_len = 0.0
    rej_len = 0.0
    for item in pair_data_ids:
        if len(item['ds_question_id'].split("@")) > 1:
            ds_question_id = '@'.join(item['ds_question_id'].split("@")[0:-1])
        else:
            ds_question_id = item['ds_question_id'].split("@")[0]
        chosen_ques_id = item['chosen']['question_id']
        reject_ques_id = item['rejected']['question_id']

        chosen_score = item['chosen']['score']
        reject_score = item['rejected']['score']
        chosen_judge = pred_quesid_to_judge[chosen_ques_id]
        reject_judge = pred_quesid_to_judge[reject_ques_id]

        chosen = question_id_2_origin_data[chosen_ques_id]
        rejected = question_id_2_origin_data[reject_ques_id]

        assert ds_question_id == str(chosen['ds_question_id']), f"{ds_question_id} != {chosen['ds_question_id']}"
        assert ds_question_id == str(rejected['ds_question_id']), f"{ds_question_id} != {rejected['ds_question_id']}"

        ch_question = chosen['question'] if 'question' in chosen else chosen['raw_question']
        rej_question = rejected['question'] if 'question' in rejected else rejected['raw_question']
        assert ch_question == rej_question, f"{chosen}\n{rejected}"

        question = ch_question
        ch_answer = chosen['answer']
        rej_answer = rejected['answer']

        ch_len += len(ch_answer.split())
        rej_len += len(rej_answer.split())

        image_path = chosen['metainfos']['image_path']
        assert chosen['metainfos']['image_path'] == rejected['metainfos']['image_path']

        if len(chosen_judge) != len([fact for fact in chosen['facts'] if fact != '']):
            print("chosen facts not match:", chosen_judge, chosen['facts'])
            continue

        if len(reject_judge) != len([fact for fact in rejected['facts'] if fact != '']):
            print("rejected facts not match:", reject_judge, rejected['facts'])
            continue

        metainfos = {
            'ds_question_id': ds_question_id,
            "reference": chosen['metainfos']['reference'] if 'reference' in chosen['metainfos'] else '',
            "origin_file": chosen['metainfos']['origin_file'] if 'origin_file' in chosen['metainfos'] else '',
            "chosen_infos": {key: chosen[key] for key in chosen if key in ['facts', 'changed_facts']},
            "rejected_infos": {key: rejected[key] for key in rejected if key in ['facts', 'changed_facts']},
            "scores": {'chosen': {'judge': chosen_judge, 'score': str(chosen_score)},
                       'rejected': {'judge': reject_judge, 'score': str(reject_score)}},
        }

        new_item = {
            "image_id": image_path.split('/')[-1],
            "image_path": image_path,
            "ds_question_id": ds_question_id,
            "question": question,
            "chosen": ch_answer,
            "rejected": rej_answer,
            "org_infos": metainfos
        }

        pair_data.append(new_item)

    print("pair data:", len(pair_data))
    print("chosen avg len:", ch_len / len(pair_data))
    print("rejected avg len:", rej_len / len(pair_data))

    # print(pair_data[0])

    write_jsonlines(save_path, pair_data)

    write_jsonlines(save_path.replace('.jsonl', '.addcls.jsonl'), pred_addcls)
    save_pred_quesid_to_judge(pred_quesid_to_judge, origin_divide_data, save_path.replace('.jsonl', '.addfactjudge.jsonl'))

    return pair_data

def sample_pair_data(pair_data, sample_num, save_path):
    dsid_2_pairs = defaultdict(list)
    for item in pair_data:
        dsid_2_pairs[item['ds_question_id']].append(item)

    sampled_pairs = []
    for key, item in dsid_2_pairs.items():
        if len(item) >= sample_num:
            sampled_items = random.sample(item, sample_num)
        else:
            sampled_items = item
        sampled_pairs += sampled_items

    print(f"sample {sample_num} pair data:", len(sampled_pairs))
    write_jsonlines(save_path, sampled_pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--autocheck_path', type=str)
    parser.add_argument('--gpt_divide_gq_path', type=str)
    parser.add_argument('--sample_num', type=int, default=2)

    args = parser.parse_args()

    path = args.autocheck_path
    path_gpt_divide = args.gpt_divide_gq_path


    save_path = path.replace('.jsonl', '.pair_diff1.jsonl')
    pair_data = get_pair_data(path, path_gpt_divide, save_path, diff=1)

    sampled_num = args.sample_num
    sample_save_path = path.replace('.jsonl', f'_pair_diff1_samp{sampled_num}.jsonl')
    pair_data = read_jsonlines(save_path)
    sample_pair_data(pair_data, sampled_num, sample_save_path)
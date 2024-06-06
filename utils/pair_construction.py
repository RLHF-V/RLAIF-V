import numpy as np
from itertools import combinations
from collections import defaultdict


def func_yes_prob(item_scores):
    yes_prob = item_scores['yes'] + item_scores['Yes']
    return yes_prob

def func_no_prob(item_scores):
    no_prob = item_scores['no'] + item_scores['No']
    return no_prob


def get_pred_scores(pred_data_addscores, func):
    pred_scores = []
    for item in pred_data_addscores:
        pred_scores.append(func(item['scores']))
    return pred_scores


def get_dsid_to_question_id(pred):
    dsid_to_question_ids = defaultdict(list)
    for item in pred:
        ds_id = item['metainfos']['ds_question_id'] if 'ds_question_id' in item['metainfos'] else item['ds_question_id']
        ques = item['metainfos']['metainfos']['origin_question']
        key = f'{ds_id}@{ques}'
        dsid_to_question_ids[key].append(item['question_id'])

    dsid_to_question_ids = {key: list(set(value)) for key, value in dsid_to_question_ids.items()}
    # print("dsid_to_question_ids:", list(dsid_to_question_ids.keys())[:10])
    # print("dsid_to_question_ids len:", len(dsid_to_question_ids))
    return dsid_to_question_ids


def pair_data_judge(data_item_0, data_item_1, diff):
    score_diff = data_item_0['score'] - data_item_1['score']
    if abs(score_diff) >= diff:
        if score_diff < 0:
            chosen = data_item_1
            rejected = data_item_0
        else:
            chosen = data_item_0
            rejected = data_item_1
        return {'chosen': chosen, 'rejected': rejected}
    else:
        return None

def get_pair_data(quesid_to_scores, dsid_to_question_ids, diff):
    pair_data = []

    for key, question_ids in dsid_to_question_ids.items():

        potential_pairs = []

        for comp_idx1, comp_idx2 in combinations(question_ids, 2):

            ans_1_score = quesid_to_scores[comp_idx1]
            ans_2_score = quesid_to_scores[comp_idx2]

            ans_1 = {"question_id": comp_idx1, "score": ans_1_score}
            ans_2 = {"question_id": comp_idx2, "score": ans_2_score}

            pair = pair_data_judge(ans_1, ans_2, diff=diff)

            if pair is not None:
                potential_pairs.append(pair)


        for chosen_pair in potential_pairs:
            chosen_pair_data = {
                "ds_question_id": key,
                "chosen": chosen_pair['chosen'],
                "rejected": chosen_pair['rejected'],
            }

            pair_data.append(chosen_pair_data)

    return pair_data


def get_pairs_inner(pred_data_addscores, diff=1, return_infos=False):
    def pred_scores_to_class(pred):
        pred_scores_yes = np.array(get_pred_scores(pred_data_addscores, func=func_yes_prob))
        pred_scores_no = np.array(get_pred_scores(pred_data_addscores, func=func_no_prob))

        pred_cls = pred_scores_yes > pred_scores_no

        pred_addcls = []
        for i,item in enumerate(pred):
            item['pred_label'] = int(pred_cls[i])
            pred_addcls.append(item)

        return pred_addcls

    def get_pred_ans_scores(pred_addcls):
        pred_quesid_to_fact_label_list = defaultdict(list)
        pred_quesid_to_judge = defaultdict(dict)
        for item in pred_addcls:
            question_id = item['question_id']
            pred_quesid_to_fact_label_list[question_id].append(item['pred_label'])
            raw_question = item['raw_question'] if 'raw_question' in item else item['question']
            pred_quesid_to_judge[question_id][raw_question] = '1' if item['pred_label']==True else '0'

        pred_quesid_to_scores = {key: sum(value) - len(value) for key, value in pred_quesid_to_fact_label_list.items()}

        return pred_quesid_to_scores, pred_quesid_to_judge

    pred_addcls = pred_scores_to_class(pred_data_addscores)
    pred_quesid_to_scores, pred_quesid_to_judge = get_pred_ans_scores(pred_addcls)

    dsid_to_question_ids = get_dsid_to_question_id(pred_data_addscores)
    pair_data = get_pair_data(pred_quesid_to_scores, dsid_to_question_ids, diff)

    if return_infos:
        return pair_data, pred_quesid_to_judge, pred_addcls
    else:
        return pair_data
import json
import os
import re
from collections import defaultdict

import numpy as np
import sys
import glob


if __name__ == '__main__':
    base_dir = sys.argv[1]
    print(base_dir)

    file_name = sys.argv[2]

    patterns = ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*']
    f_list = sum([list(glob.glob(base_dir + p)) for p in patterns], [])
    review_files = [x for x in f_list if x.endswith(
        '.json') and file_name in x and 'prev' not in x]

    model_results = {}
    for review_file in sorted(review_files):
        data = json.load(open(review_file))
        if 'checkpoint-' not in review_file:
            *_, model, _, _ = review_file.split('/')
            # print(re.findall(r'_{}step-', model))
            step = int(re.findall(r'_(\d+)step', model)[0])
            # print(step)
        else:
            *_, model, _, step, _ = review_file.split('/')
            step = int(step.split('-')[-1])

        if model not in model_results:
            model_results[model] = defaultdict(dict)

        metrics = data['overall_metrics']
        model_results[model][step] = metrics

    for model in model_results:
        steps = sorted(model_results[model])
        print(f'\n===> {model}')
        for step in steps:
            metrics = model_results[model][step]
            correct_response = metrics['correct_rate'] * 100
            obj_correct_rate = metrics['object_correct_rate'] * 100
            obj_recall = metrics['obj_rec'] * 100
            coco_sentence_num = metrics['coco_sentence_num']
            coco_word_count = metrics['coco_word_count']
            gt_word_count = metrics['gt_word_count']
            avg_length = metrics['avg_word_len']
            sent_num = metrics['sentence_num']

            obj_f1 = 2 * obj_recall * obj_correct_rate / \
                (obj_recall + obj_correct_rate)
            res_f1 = 2 * (coco_sentence_num / 3) * correct_response / \
                (coco_sentence_num / 3 + correct_response)

            print(f'{step:3d}\t{correct_response:.2f}\t{obj_correct_rate:.2f}\t{obj_recall:.2f}\t{obj_f1:.2f}\t{res_f1:.2f}\t{avg_length:.2f}\t{coco_sentence_num}\t{sent_num}\t{coco_word_count}\t{gt_word_count}')

            # print(f'Response Correct: {correct_response:.2f}\n'
            #       f'Object Correct  : {obj_correct_rate:.2f}\n'
            #       f'Object Recall   : {obj_recall:.2f}\n'
            #       f'Average Length  : {avg_length:.2f}\n'
            #       f'COCO Sent Number: {coco_sentence_num}\n'
            #       f'COCO Word Number: {coco_word_count}\n'
            #       f'GT Word Number  : {gt_word_count}')

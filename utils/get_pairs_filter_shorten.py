import argparse
import pandas as pd
import numpy as np
from file_io import read_jsonlines, write_jsonlines

def filter_pair_by_len(all_pairs_dicts, diff_len):
    remain_pair = []
    for pair in all_pairs_dicts:
        chosen_len = len(pair['chosen'].split())
        reject_len = len(pair['rejected'].split())
        pair_diff_len = (reject_len - chosen_len)/float(reject_len)
        if pair_diff_len > diff_len:
            continue

        if pair['chosen'].strip() == pair['rejected'].strip():
            print(diff_len, "chosen==rejected")
            continue

        remain_pair.append(pair)

    return remain_pair

def cal_pair_statistics(all_pairs):
    avg_win_len = 0.0
    avg_lose_len = 0.0
    shorten_cnt = 0
    longer_cnt = 0

    total = 0

    for i in range(len(all_pairs)):
        data = all_pairs[i]

        avg_win_len += len(data['chosen'].split())
        avg_lose_len += len(data['rejected'].split())

        if len(data['chosen'].split()) > len(data['rejected'].split()):
            longer_cnt += 1
        elif len(data['chosen'].split()) < len(data['rejected'].split()):
            shorten_cnt += 1

        total += 1

    avg_win_len /= total
    avg_lose_len /= total
    shorten_cnt /= total
    longer_cnt /= total
    return avg_win_len, avg_lose_len, shorten_cnt, longer_cnt

def cal_pair_search_difflen(wanted_pairs_dicts, use_len=True):
    shorten_ratios = np.arange(0, 1, 0.02)

    results = []
    for ratio in shorten_ratios:
        remain_pairs = filter_pair_by_len(wanted_pairs_dicts, ratio)
        # print("filtered short:", len(remain_pairs))
        avg_win_len, avg_lose_len, shorten_cnt, longer_cnt = cal_pair_statistics(remain_pairs)

        avg_diff_len = abs(avg_win_len - avg_lose_len) / avg_lose_len
        diff_shorter_longer_portion = abs(shorten_cnt - longer_cnt)
        results.append({
            'ratio': ratio,
            'avg_win_len': avg_win_len,
            'avg_lose_len': avg_lose_len,
            'shorten_portion': shorten_cnt,
            'longer_portion': longer_cnt,
            'avg_diff_len': abs(avg_win_len - avg_lose_len),
            'avg_diff_len_portion': avg_diff_len,
            'diff_shorter_longer_portion': diff_shorter_longer_portion,
            'total_diff_portion': avg_diff_len if use_len else diff_shorter_longer_portion  # + diff_shorter_longer_portion
        })

    df = pd.DataFrame(results)
    # print(df)

    idmin = df['total_diff_portion'].idxmin()
    # print(df.iloc[idmin])

    final_remain_pairs = filter_pair_by_len(wanted_pairs_dicts, df.iloc[idmin]['ratio'])
    return final_remain_pairs, df.iloc[idmin]['ratio'], df.iloc[idmin], df

def main_filter_shorten(path, save_path, use_len=True):
    all_pairs_dicts = read_jsonlines(path)
    final_remain_pairs, ratio, df_min, df = cal_pair_search_difflen(all_pairs_dicts, use_len=use_len)
    print(ratio)
    print(df_min)
    write_jsonlines(save_path, final_remain_pairs)
    df_min.to_excel(save_path.replace('.jsonl', '_search_min_diff_statistics.xlsx'))
    df.to_excel(save_path.replace('.jsonl', '_search_diff.xlsx'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()

    path = args.path
    save_path = args.save_path

    main_filter_shorten(
        path,
        save_path
    )
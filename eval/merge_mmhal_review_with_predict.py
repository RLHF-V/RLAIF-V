import json
import jsonlines
import argparse

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data

def merge_reviews(review_path, predict_path, save_path):
    with open(review_path, "r") as f:
        reviews = json.load(f)

    try:
        with open(predict_path, "r") as f:
            predicts = json.load(f)
    except:
        predicts = read_jsonl(predict_path)


    for i in range(len(predicts)):
        predicts[i]["gpt4_review"] = reviews[i]["choices"][0]["message"]["content"]

    with open(save_path, "w") as f:
        json.dump(predicts, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_path', type=str)
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    print("======= merge review =========")
    print(args)

    merge_reviews(args.review_path, args.predict_path, args.save_path)
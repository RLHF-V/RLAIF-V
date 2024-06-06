import io
import os
import json
import copy
import torch
import base64
import argparse

import math
import tqdm
import torch.utils.data as torch_data

from PIL import Image
from transformers import AutoModel, AutoTokenizer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class MiniCPMVQADataset(torch_data.Dataset):
    def __init__(self, qa_file, max_size=-1, start=0, end=-1, repeat=1, chunk_num=1, chunk_idx=0):
        '''
        qa_file: jsonl file that each line is a dict like {
            'image': b64img,
            'question': question_text
        }
        '''
        super().__init__()

        self.qa_file = qa_file
        try:
            self.qa_data = [json.loads(line) for line in open(self.qa_file)]
            if isinstance(self.qa_data[0], list):
                # unwrap one-line json question file
                self.qa_data = self.qa_data[0]
        except:
            try:
                with open(self.qa_file, "r") as f:
                    self.qa_data = json.load(f)
            except:
                raise ValueError("Wrong input data format!")

        new_qa_data = []
        for item in self.qa_data:
            new_qa_data += [item] * repeat
        self.qa_data = new_qa_data

        new_qa_data = []
        if 'question_id' not in self.qa_data[0].keys():
            for idx,item in enumerate(self.qa_data):
                new_item = copy.deepcopy(item)
                new_item['question_id'] = idx
                new_qa_data.append(new_item)
            self.qa_data = new_qa_data
        # print("repeat:", self.qa_data[0]['question_id'], self.qa_data[1]['question_id'], self.qa_data[4]['question_id'])

        start = start*repeat
        end = end*repeat
        if end < 0 or end > len(self.qa_data):
            self.qa_data = self.qa_data[start:]
        else:
            self.qa_data = self.qa_data[start:end]

        print("final qa data len:", len(self.qa_data), f"\nstart={start} end={end}")

        # print("before chunk:", self.qa_data[0]['question_id'], self.qa_data[1]['question_id'], self.qa_data[4]['question_id'])
        self.qa_data = get_chunk(self.qa_data, chunk_num, chunk_idx)
        print("chunk_num:", chunk_num, "chunk_idx:", chunk_idx, "data_len:", len(self.qa_data))

    def __getitem__(self, index):
        item = self.qa_data[index]
        # print(item['question_id'], flush=True)
        if "image_id" in item.keys():
            imgid = item["image_id"]
        else:
            imgid = ''

        if "image" in item.keys():
            img_b64 = item['image']

            if len(img_b64) > 100:
                image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
            else:
                image = Image.open(img_b64).convert('RGB')
        elif "image_path" in item.keys():
            print("in path")
            image = Image.open(item['image_path']).convert('RGB')
        elif "image_path" in item['metainfos'].keys():
            print("in metainfos")
            image = Image.open(item['metainfos']['image_path']).convert('RGB')

        metainfo = {key: value for key, value in item.items() if key not in [
            "image_id", "question", "image"]}

        raw_question = item['question']
        question = [{"role": "user", "content": raw_question}]

        return {
            'image': image,
            'raw_question': raw_question,
            'question': question,
            'image_id': imgid,
            'metainfo': metainfo,
            'question_id': item['question_id'] if 'question_id' in item else index,
            'origin_dataset': self.qa_file,
        }

    def __len__(self):
        return len(self.qa_data)


class MiniCPM_Llama3_V:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input, sampling=False, temperature=0.7, max_tokens=512):
        image = input['image']
        msgs = input['question']

        answer = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=sampling,
            temperature=temperature,
            max_new_tokens=max_tokens
    	)
        return answer


def eval_model(args):
    model = MiniCPM_Llama3_V(args.model_name)

    answer_dir = '/'.join(args.answers_file.split("/")[:-1])
    print(answer_dir)
    assert os.path.exists(answer_dir), f'Expecting {answer_dir} to be existing'
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    qa_dataset = MiniCPMVQADataset(args.question_file, start=args.start, end=args.end, repeat=args.repeat, chunk_num=args.chunk_num, chunk_idx=args.chunk_idx)

    ans_file = open(answers_file, "w")

    with torch.inference_mode():
        for batch in tqdm.tqdm(qa_dataset, f'Generating answers'):
            response = model.chat(batch, sampling=args.sampling, temperature=args.temperature, max_tokens=args.max_tokens) # temperature=args.temperature

            if 'ds_question_id' in batch['metainfo']:
                ans_file.write(json.dumps({
                    'question_id': batch['question_id'],
                    'ds_question_id': batch['metainfo']['ds_question_id'],
                    'raw_question': batch['raw_question'],
                    'answer': response,
                    'metainfos': batch['metainfo'],
                    'model_path': args.model_name
                }) + "\n")
            else:
                ans_file.write(json.dumps({
                    'question_id': batch['question_id'],
                    'raw_question': batch['raw_question'],
                    'answer': response,
                    'metainfos': batch['metainfo'],
                    'model_path': args.model_name
                }) + "\n")

            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sampling", action='store_true')
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--chunk-num", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    eval_model(args)
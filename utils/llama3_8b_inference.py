import os
import math
import copy
import json
import torch
import argparse
import transformers
from tqdm import tqdm

import torch.utils.data as torch_data


class GenDataset(torch_data.Dataset):
    def __init__(self, data, wrap_func, pipline):
        super().__init__()
        self.data = data
        self.wrap_func = wrap_func
        self.pipeline = pipline

    def __getitem__(self, index):
        item = self.data[index]
        messages = self.wrap_func(item)

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "batch_data": item,
            "prompt": prompt
        }

    def __len__(self):
        return len(self.data)

def data_collate_fn(data_list):
    batch_data = [x['batch_data'] for x in data_list]
    prompts = [x['prompt'] for x in data_list]

    data = {
        'batch_data': batch_data,
        'prompt': prompts,
    }

    return data

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonlines(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data
def write_jsonlines(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def get_facts(result):
    result = result.strip().split('\n')

    fact_list = []
    for item in result:
        if item == '':
            continue
        if '###' in item:
            continue

        item = item[1:].strip()
        fact_list.append(item)

    # print(fact_list)
    return fact_list

def init_divide_pipline():
    model_id = "/data/apps/RLAIF-V/rlaif-v-main/models/llama3-split"
    tokenizer = (model_id, {'padding_side': 'left'})
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        tokenizer=tokenizer
    )
    return tokenizer, pipeline


def init_changeq_pipline():
    model_id = "/data/apps/RLAIF-V/rlaif-v-main/models/llama3-changeq"
    tokenizer = (model_id, {'padding_side': 'left'})
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        tokenizer=tokenizer
    )
    return tokenizer, pipeline

def batch_inference(path, ans_file, tokenizer, pipeline, key, wrap_func, batch_size=8, chunk_num=1, chunk_idx=0, max_tokens=140, start=0, end=-1):
    # load data
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        data = []
        for line in open(path, 'r'):
            data.append(json.loads(line))

    if type(data) == dict:
        data = [data]

    print(f"start={start}, end={end}")
    if end > 0:
        end = min(end, len(data))
    elif end == -1:
        end = len(data)

    data = data[start:end]

    # get current chunk data
    data = get_chunk(data, chunk_num, chunk_idx)

    # load prev inference results
    if os.path.exists(ans_file):
        prev_ans = []
        with open(ans_file, 'r') as f:
            for line in f.readlines():
                temp_ans = json.loads(line)
                prev_ans.append(temp_ans)

        ans_f = open(ans_file, 'a')
        data = data[len(prev_ans):]
    else:
        prev_ans = []
        os.makedirs(os.path.dirname(ans_file), exist_ok=True)
        ans_f = open(ans_file, 'w')

    # get batch inputs
    dataset = GenDataset(data, wrap_func, pipeline)
    print(f'Dataset size is {len(dataset)}')
    print(f'Dataset batch size is {batch_size}')

    dataloader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=5,
        pin_memory=True,
        drop_last=False,
        collate_fn=data_collate_fn,
    )
    print(f'Dataloader size is {len(dataloader)}')

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    # inference
    all_outputs = copy.deepcopy(prev_ans)
    for i, batch_list in tqdm(enumerate(dataloader)):
        outputs = pipeline(
            batch_list['prompt'],
            eos_token_id=terminators,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
            # num_beams=3,
            top_p=0.9,
            batch_size=batch_size,
        )

        for item, prompt, output in zip(batch_list['batch_data'], batch_list['prompt'], outputs):
            resp = output[0]["generated_text"][len(prompt):]

            item[f'raw_{key}'] = resp
            item[key] = get_facts(resp)

            all_outputs.append(item)

            ans_f.write(json.dumps(item, ensure_ascii=False) + '\n')
            ans_f.flush()

    ans_f.close()

    return all_outputs

def wrap_prompt_divide_to_list(item):
    if 'raw_question' in item.keys():
        question = item['raw_question']
    elif 'prompt' in item.keys():
        question = item['prompt']
    else:
        question = item['question']
    answer = item['answer'] if 'answer' in item.keys() else item['text']

    content="You are an expert in extracting facts from the given question-answer pair for an image. Your task is to extract and rewrite the facts mentioned in the question-answer pair into self-contained sentences. Exclude opinions or subjective statements.\n\nYou should present your result in the following format:\n### Facts:\n- {Extracted fact 1}\n- {Extracted fact 2}\n- ...\n\n### Question-answer pair:\nQuestion: " + question + "\nAnswer: " + answer
    temp_input = ' '.join(content.split(' ')[:300])

    messages = [{"role": "user", "content": temp_input},]
    return messages

def wrap_prompt_changeq_to_list(item):
    facts=item["facts"]
    content="You are an expert at modifying a given declarative sentence into a general question sentence. Your task is to modify the given declarative sentences one by one into a general question form. Do not change tenses or add extra content.\n    If the given declarative sentence contains not, no or negative meaning words, you need to check the modified general interrogative sentence to make sure that the generated general question sentence retains words with not , no or negative meaning words.\n\nYou should present your result in the following format:\n### Modified sentences:\n- {Modified sentence 1}\n- {Modified sentence 2}\n- ...\n\n### Declarative sentences:"
    for fact in facts:
        content+="\n- {}\n".format(fact)

    messages = [
        {"role": "user", "content": content},
    ]
    return messages

def data_collator(data, pipeline, wrap_func, batch_size=8):
    prompt_list = []
    for item in data:
        messages = wrap_func(item)

        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_list.append(prompt)

    prompt_list = [prompt_list[i:i+batch_size] for i in range(0, len(prompt_list), batch_size)]
    origin_data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    assert len(prompt_list) == len(origin_data)

    batch_inputs = []
    for i in range(len(prompt_list)):
        batch_inputs.append(
            {"batch_data": origin_data[i], "prompt": prompt_list[i]}
        )
    return batch_inputs

def construct_question_yesno(path, save_path):
    print("construct_question_yesno")
    data = read_jsonlines(path)

    new_qas = []
    for i,item in enumerate(data):
        try:
            image_path = item['image_path']
        except:
            try:
                image_path = item['metainfos']['image_path']
            except:
                raise ValueError("Do not have 'image_path' in the data!")

        if type(item['facts']) == type(''):
            continue

        for fact,changed_fact in zip(item['facts'],item['changed_facts']):
            question = f'{changed_fact} Please answer yes or no.'

            metainfos = copy.deepcopy(item['metainfos'])
            metainfos['origin_question'] = item['raw_question'] if 'raw_question' in item else item['question']
            metainfos['origin_answer'] = item['answer']
            metainfos['origin_fact'] = fact
            metainfos['origin_changed_fact']=changed_fact
            metainfos['origin_all_facts'] = item['facts']
            metainfos['origin_changed_all_facts']=item['changed_facts']

            new_item = {
                'question_id': item['question_id'],
                'ds_question_id': item['ds_question_id'] if 'ds_question_id' in item else item['metainfos']['ds_question_id'],
                'image_path': image_path,
                'question': question,
                'metainfos': metainfos
            }

            new_qas.append(new_item)

    write_jsonlines(save_path, new_qas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--divide_suffix', type=str, default='llama3-8b_divide')
    parser.add_argument('--chunk-num', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()

    print(f"chunk_num={args.chunk_num}, chunk_idx={args.chunk_idx}")
    tokenizer, pipeline = init_divide_pipline()

    path = args.path

    save_divide_path = path.replace('.jsonl', f'.s{args.start}-e{args.end}.chunk{args.chunk_num}-{args.chunk_idx}.{args.divide_suffix}.jsonl')
    print(f"==> Do split... \n save path = {save_divide_path}")
    all_outputs = batch_inference(path, save_divide_path, tokenizer, pipeline, key='facts', wrap_func=wrap_prompt_divide_to_list,
                                  batch_size=args.bs, chunk_num=args.chunk_num, chunk_idx=args.chunk_idx, max_tokens=256,
                                  start=args.start, end=args.end)
    del tokenizer
    del pipeline

    tokenizer, pipeline = init_changeq_pipline()
    save_general_questions_path=save_divide_path.replace(".jsonl", ".gq.jsonl")
    print(f"\n==> Do change question... \n save path = {save_general_questions_path}")
    all_outputs = batch_inference(save_divide_path, save_general_questions_path, tokenizer, pipeline,
                                  key='changed_facts', wrap_func=wrap_prompt_changeq_to_list,
                                  batch_size=args.bs, max_tokens=256)

    save_q_path = save_general_questions_path.replace(".jsonl", ".qas.jsonl")
    construct_question_yesno(save_general_questions_path, save_q_path)

if __name__ == "__main__":
    main()
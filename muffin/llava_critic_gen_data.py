import copy
import datetime
import os
import io
import warnings

import datasets
import numpy as np
import pandas as pd
import tqdm
import json
import base64
import random
import argparse
import itertools
from PIL import Image
from functools import partial

import torch
import torch.utils.data as torch_data

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from builder.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from muffin.gen_data_util import InferenceSampler, torch_pad_sequence


class GenDataset(torch_data.Dataset):
    __critic_prompt = DEFAULT_IMAGE_TOKEN + "\n" + "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [{}]\nThe first response: [{}]\nThe second response: [{}]\nASSISTANT:\n"

    def __init__(self, file, question_process):
        '''
        qa_file: jsonl file that each line is a dict like {
            'image': b64img,
            'question': question_text
        }
        '''
        super().__init__()
        self.file = file
        self.data = datasets.load_dataset(self.file, cache_dir='./cache')['train'].cast_column(
            'image',
            datasets.Image(decode=False)
        )
        self.question_process = question_process

        # Build grouped data by 'idx' for prompt construction
        self.grouped_data = self._group_data_by_idx()
        # Build critic prompts
        self.critic_prompts = self._build_critic_prompts()

    def _group_data_by_idx(self):
        """
        Groups the data by 'idx' field to prepare for prompt construction.
        """
        grouped = {}
        for item in self.data:
            idx = item.get('idx')
            if idx is None:
                raise KeyError("Each data item must have an 'idx' field.")
            if idx not in grouped:
                grouped[idx] = []
            grouped[idx].append(item)
        print(f"Grouped data into {len(grouped)} groups based on 'idx' field.")
        return grouped

    def _build_critic_prompts(self):
        """
        Constructs critic prompts for each pair of responses within the same group.
        """
        res = []
        for idx, items in self.grouped_data.items():
            for i, pair in enumerate(items):
                question = pair.get('question')
                response_1 = pair.get('chosen')
                if question is None or response_1 is None:
                    raise KeyError("Each data item must have 'question' and 'chosen' fields.")

                for j, compare in enumerate(items):
                    if j == i:
                        continue
                    response_2 = compare.get('chosen')
                    if response_2 is None:
                        raise KeyError("Each data item must have a 'chosen' field for comparison.")

                    prompt = self.__critic_prompt.format(question, response_1, response_2)
                    res.append({
                        'idx': idx,
                        'inner_idx': i,
                        'question_id': pair.get('question_id', idx),
                        'raw_question': prompt,
                        'image': pair['image'],
                        'metainfos': {k: v for k, v in pair.items() if
                                      k not in ["image_id", "question", "image", "chosen"]},
                        'origin_dataset': self.file,
                    })
        print("Completed building critic prompts for all pairs.")
        return res

    def __getitem__(self, index):
        item = self.critic_prompts[index]

        # Process image
        if "image" in item.keys():
            img = item['image']['bytes']
            image = Image.open(io.BytesIO(img)).convert('RGB')
        elif "image_path" in item.keys():
            image = Image.open(item['image_path']).convert('RGB')
        elif "image_path" in item['metainfos'].keys():
            image = Image.open(item['metainfos']['image_path']).convert('RGB')
        else:
            raise ValueError("Unable to read image")

        metainfo = {key: value for key, value in item.items() if key not in ["image_id", "question", "image", "chosen"]}

        raw_question = item['raw_question']

        question_input_ids = self.question_process(raw_question)
        # print("question_input_ids:", question_input_ids)

        return {
            'idx': item['idx'],
            'inner_idx': item['inner_idx'],
            'question_id': item['question_id'] if 'question_id' in item else index,
            'image': image,
            'question_input_ids': question_input_ids,
            'raw_question': raw_question,
            'metainfos': metainfo,
            'origin_dataset': self.file,
        }

    def __len__(self):
        return len(self.critic_prompts)


def wrap_question_for_llava15(question, tokenizer):
    if isinstance(question, list):
        question = question[0]
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    return input_ids


def llava15_qa_colloator_fn(data_list, tokenizer, image_processor, config):
    idx = [x['idx'] for x in data_list]
    inner_idx = [x['inner_idx'] for x in data_list]
    input_ids = [torch.as_tensor(x['question_input_ids']) for x in data_list]

    input_ids = torch_pad_sequence(
        input_ids, tokenizer.pad_token_id, padding_side='left')

    # images = [process_images([x['image']], image_processor, config)[0].to(dtype=torch.float16) for x in data_list]
    images = process_images([x['image'] for x in data_list], image_processor, config)
    images = [_image.to(dtype=torch.float16) for _image in images]

    image_sizes = [x['image'].size for x in data_list]

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
        'idx': idx,
        'inner_idx': inner_idx,
        'images': images,
        'image_sizes': image_sizes,
        'input_ids': input_ids,
        'raw_questions': raw_questions,
    }

    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]
    if 'origin_dataset' in data_list[0]:
        data['origin_dataset'] = [x['origin_dataset'] for x in data_list]
    if 'answer' in data_list[0]:
        data['gt_answers'] = [x['answer'] for x in data_list]
    if 'image_id' in data_list[0]:
        data['image_id'] = [x['image_id'] for x in data_list]
    if 'metainfo' in data_list[0]:
        data['metainfo'] = [x['metainfo'] for x in data_list]
    if 'metainfos' in data_list[0]:
        data['metainfos'] = [x['metainfos'] for x in data_list]

    return data


def serialize_metainfos(metainfos):
    if isinstance(metainfos, dict) or isinstance(metainfos, list):
        return json.dumps(metainfos)
    return ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--ds_name', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_beam', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=10)
    parser.add_argument('--answer_dir', type=str)

    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
            timeout=datetime.timedelta(days=2)
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print(f'Init Rank-{rank} / World Size-{world_size}')

    model_path = os.path.expanduser(args.checkpoint)
    model_name = "llava_qwen"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map={"": 'cuda'})

    random.seed(args.seed)

    question_process_func = partial(
        wrap_question_for_llava15, tokenizer=tokenizer)

    dataset = GenDataset(args.ds_name, question_process_func)
    print(f'Dataset size is {len(dataset)}')

    collate_fn = partial(llava15_qa_colloator_fn, tokenizer=tokenizer,
                         image_processor=image_processor, config=model.config)
    dataloader = torch_data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f'Dataloader size is {len(dataloader)}')

    outputs = []
    force_sync = 0
    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, f'Generating answers'):
            batch['images'] = [img.cuda(non_blocking=True) for img in batch['images']]
            if args.num_beam >= 1:
                output = model.generate(
                    batch['input_ids'].cuda(),
                    images=batch['images'],
                    image_sizes=batch['image_sizes'],
                    do_sample=False,
                    num_beams=args.num_beam,
                    max_new_tokens=args.max_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    modalities=["image"] * args.batch_size
                )
            else:
                output = model.generate(
                    batch['input_ids'].cuda(),
                    images=batch['images'],
                    image_sizes=batch['image_sizes'],
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    modalities=["image"] * args.batch_size)

            for question, output_ids, question_id, metainfos, idx, inner_idx in zip(
                    batch['raw_questions'],
                    output.sequences.cpu(),
                    batch['question_id'],
                    batch['metainfos'],
                    batch['idx'],
                    batch['inner_idx']):
                response = tokenizer.decode(
                    output_ids, skip_special_tokens=True)
                response = response.strip()
                # The better answer: [second]
                # The better answer: [first]
                # Two responses are equally good.

                serialized_metainfos = serialize_metainfos(metainfos)

                outputs.append({
                    'idx': idx,
                    'inner_idx': inner_idx,
                    'question_id': question_id,
                    'raw_question': question,
                    'answer': response,
                    'metainfos': serialized_metainfos,
                    'model_path': args.checkpoint,
                })

            del batch
            del output
            torch.cuda.empty_cache()
            force_sync += 1
            if force_sync == 50:
                torch.distributed.barrier()
                force_sync = 0

    print(len(outputs))

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]

    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    print(f'Merged outputs: {len(merged_outputs)}')

    if torch.distributed.get_rank() == 0:
        step = 5000
        for idx, start in enumerate(range(0, len(merged_outputs), step)):
            temp_data = merged_outputs[start: min(start + step, len(merged_outputs))]
            # Verify data before creating DataFrame
            if len(temp_data) == 0:
                continue  # Skip empty batches

            # Optionally, serialize metainfos if not already done
            for entry in temp_data:
                if isinstance(entry['metainfos'], dict):
                    entry['metainfos'] = json.dumps(entry['metainfos'])

            df = pd.DataFrame(temp_data)
            output_file = os.path.join(
                args.answer_dir,
                f'RLAIF-V-Dataset-llava-critic_{idx:03}-{len(temp_data)}.parquet'
            )
            temp_file = output_file + '.tmp'
            df.to_parquet(temp_file)
            os.rename(temp_file, output_file)
            print(f"Saved {len(temp_data)} records to {output_file}")

    torch.distributed.barrier()

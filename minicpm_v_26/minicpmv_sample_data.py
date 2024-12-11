import multiprocessing
import os
import random
from functools import partial

import torch
import torch.utils.data as torch_data
from transformers import AutoProcessor
import numpy as np

from muffin.gen_data_util import InferenceSampler, torch_pad_sequence
from muffin.sample_data_util import SampleDataset, sample_and_record
from builder.builder import load_pretrained_model


def collactor_fn(data_list, processor):
    prompt_list = [x['question_input_ids'] for x in data_list]
    image_list = [[x['image']] for x in data_list]

    data = processor(prompt_list, image_list, return_tensors="pt", max_length=8192).to(torch.device("cuda"))
    # data = {
    #     "input_ids": processed["input_ids"],
    #     "attention_mask": processed["attention_mask"],
    #     "pixel_values": processed["pixel_values"],
    #     "image_sizes": processed["image_sizes"],
    #     "image_bound": processed["image_bound"],
    #     "tgt_sizes": processed["tgt_sizes"]
    # }
    # data = {key: value.to(torch.device("cuda")) for key, value in data.items()}

    data['raw_questions'] = [x['raw_question'] for x in data_list]
    data['raw_images'] = [x['raw_image'] for x in data_list]

    if 'question_id' in data_list[0]:
        data['question_id'] = [x['question_id'] for x in data_list]
    if 'idx' in data_list[0]:
        data['idx'] = [x['idx'] for x in data_list]
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

def wrap_question(question, tokenizer):
    pattern = "(<image>./</image>)"
    if "<image>" in question:
        question = question.replace("<image>", pattern)
    else:
        question = f"{pattern}{question}"
    msgs_list = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs_list, tokenize=False)


def main(model_name, model_path, ds_path, answer_dir, sample=10, seed=0, batch_size=10,
         num_workers=16, max_tokens=512, temperature=0.7):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

        print(f'Init Rank-{torch.distributed.get_rank()}')
    multiprocessing.set_start_method('spawn')
    random.seed(seed)
    np.random.seed(seed)

    tokenizer, model, _, _ = load_pretrained_model(model_path, None, model_name)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    question_process_fc = partial(wrap_question, tokenizer=tokenizer)
    dataset = SampleDataset(ds_path, question_process_fc, repeat_time=sample)
    print(f'Dataset size is {len(dataset)}')

    collate_fn = partial(collactor_fn, processor=processor)
    dataloader = torch_data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=batch_size,
        # num_workers=num_workers,
        # pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f'Dataloader size is {len(dataloader)}')

    sample_and_record(dataloader, model_path, model, tokenizer, answer_dir, temperature, max_tokens)
    del model

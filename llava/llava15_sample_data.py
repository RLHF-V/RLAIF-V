import os
import random
from functools import partial

import torch
import torch.utils.data as torch_data

from builder.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path
from muffin.gen_data_util import InferenceSampler, torch_pad_sequence
from muffin.sample_data_util import SampleDataset, sample_and_record
from muffin.llava15_gen_data import wrap_question_for_llava15


def llava15_colloator_fn(data_list, tokenizer, image_processor, config):
    input_ids = [torch.as_tensor(x['question_input_ids']) for x in data_list]

    input_ids = torch_pad_sequence(
        input_ids, tokenizer.pad_token_id, padding_side='left')

    # NOTE: here we need to pass `raw_images`
    images = [process_images([x['image']], image_processor, config)[0] for x in data_list]
    images = torch.stack(images)
    raw_images = [x['raw_image'] for x in data_list]

    image_sizes = [x['image'].size for x in data_list]

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
        'images': images,
        'image_sizes': image_sizes,
        'input_ids': input_ids,
        'raw_questions': raw_questions,
        'raw_images': raw_images,
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


def main(model_name, model_path, model_base, ds_path, answer_dir, sample=10, seed=0, batch_size=10,
         num_workers=16, conv_mode='llava_v1', max_tokens=512, temperature=0.7):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl',
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            rank=int(os.getenv('RANK', '0')),
        )
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
        print(f'Init Rank-{torch.distributed.get_rank()}')

    model_path = os.path.expanduser(model_path)
    if model_name is None:
        model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                           device_map={
                                                                               "": 'cuda'})  # device_map={"": 'cuda'}

    random.seed(seed)

    question_process_func = partial(
        wrap_question_for_llava15, tokenizer=tokenizer, mm_use_im_start_end=model.config.mm_use_im_start_end,
        conv_mode=conv_mode)

    dataset = SampleDataset(ds_path, question_process_func, repeat_time=sample)
    print(f'Dataset size is {len(dataset)}')

    collate_fn = partial(llava15_colloator_fn, tokenizer=tokenizer,
                         image_processor=image_processor, config=model.config)
    dataloader = torch_data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print(f'Dataloader size is {len(dataloader)}')

    sample_and_record(dataloader, model_path, model, tokenizer, answer_dir, temperature, max_tokens)
    del model

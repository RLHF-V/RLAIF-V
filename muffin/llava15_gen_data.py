import os
import io
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
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def torch_pad_sequence(sequence, padding_value, batch_first=True, padding_side='right'):

    if padding_side == 'right':
        sequence = torch.nn.utils.rnn.pad_sequence(
            sequence,
            batch_first=batch_first,
            padding_value=padding_value)
    elif padding_side == 'left':
        sequence = torch.nn.utils.rnn.pad_sequence(
            [v.flip(-1) for v in sequence],
            batch_first=batch_first,
            padding_value=padding_value)
        sequence = sequence.flip(-1)
    else:
        raise NotImplementedError(f'padding_size={padding_side}')
    return sequence

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

class GenDataset(torch_data.Dataset):
    def __init__(self, qa_file, question_process, max_size, start=0, end=-1, repeat_time=1):
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
                self.qa_data = self.qa_data[0] # unwrap one-line json question file
        except:
            try:
                with open(self.qa_file, "r") as f:
                    self.qa_data = json.load(f)
            except:
                raise ValueError("Wrong input data format!")

        if end < 0 or end > len(self.qa_data):
            self.qa_data = self.qa_data[start:]
        else:
            self.qa_data = self.qa_data[start:end]

        print("org qa data len:", len(self.qa_data), f"\nstart={start} end={end}")

        if max_size == -1:
            max_size = len(self.qa_data)
        self.max_size = max_size

        print("max_size:", self.max_size)

        self.line_numbers = list(range(max_size))

        self.qa_data = [self.qa_data[i] for i in self.line_numbers]

        new_qa_data = []
        for item in self.qa_data:
            new_qa_data += [item] * repeat_time

        self.qa_data = new_qa_data
        print("final qa data len:", len(self.qa_data))

        self.question_process = question_process

        self.start_idx = repeat_time * start

    def __getitem__(self, index):
        item = self.qa_data[index]
        if "image_id" in item.keys():
            imgid = item["image_id"]

        # print(item.keys())
        if "image" in item.keys():
            img_b64 = item['image']

            if len(img_b64) > 100:
                image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
            else:
                image = Image.open(img_b64).convert('RGB')
        elif "image_path" in item.keys():
            # print("in")
            image = Image.open(item['image_path']).convert('RGB')
        elif "image_path" in item['metainfos'].keys():
            # print("in metainfos")
            image = Image.open(item['metainfos']['image_path']).convert('RGB')

        metainfo = {key:value for key,value in item.items() if key not in ["image_id", "question", "image"]}

        raw_question = item['question']

        question_input_ids = self.question_process(raw_question)
        # print("question_input_ids:", question_input_ids)

        return {
            'question_id': item['question_id'] if 'question_id' in item else self.start_idx+index,
            'image': image,
            'question_input_ids': question_input_ids,
            'raw_question': raw_question,
            'metainfos': metainfo,
            'origin_dataset': self.qa_file
        }

    def __len__(self):
        return len(self.qa_data)

def wrap_question_for_llava15(question, tokenizer, mm_use_im_start_end, conv_mode):
    qs = question
    if DEFAULT_IMAGE_TOKEN in qs:
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, '')

    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return input_ids

def llava15_qa_colloator_fn(data_list, tokenizer, image_processor, config):
    input_ids = [torch.as_tensor(x['question_input_ids']) for x in data_list]

    input_ids = torch_pad_sequence(
        input_ids, tokenizer.pad_token_id, padding_side='left')
    # print("input_ids:", input_ids)

    images = [process_images([x['image']], image_processor, config)[0] for x in data_list]
    images = torch.stack(images)

    image_sizes = [x['image'].size for x in data_list]

    raw_questions = [x['raw_question'] for x in data_list]
    data = {
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    # TextVQA, DocVQA, OCRVQA, STVQA
    parser.add_argument('--ds_name', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_sample', type=int, default=-1)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_beam', type=int, default=-1)
    parser.add_argument('--max_tokens', type=int, default=10)
    parser.add_argument('--answer_file', type=str)
    parser.add_argument(
        "--is_yesno",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    print(f'Init Rank-{torch.distributed.get_rank()}')
    model_path = os.path.expanduser(args.checkpoint)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map={"": 'cuda'}) # device_map={"": 'cuda'}

    random.seed(args.seed)

    question_process_func = partial(
            wrap_question_for_llava15, tokenizer=tokenizer, mm_use_im_start_end=model.config.mm_use_im_start_end, conv_mode=args.conv_mode)

    dataset = GenDataset(args.ds_name, question_process_func, max_size=args.max_sample, start=args.start_pos, end=args.end_pos, repeat_time=args.repeat)
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

    yes_id = tokenizer.encode('<s> yes')[-1]
    Yes_id = tokenizer.encode('<s> Yes')[-1]
    no_id = tokenizer.encode('<s> no')[-1]
    No_id = tokenizer.encode('<s> No')[-1]

    outputs = []
    cnt = 0
    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, f'Generating answers'):
            input_size = batch['input_ids'].shape[-1]
            # print(f'input_ids: {batch["input_ids"]}')
            # print(f'Input: {tokenizer.batch_decode(batch["input_ids"])}'
            #       f'input_ids: {batch["input_ids"]}')
                #   f'attn_mask: {batch["attention_mask"]}')
            if args.is_yesno:
                output = model.generate(
                    inputs=batch['input_ids'].cuda(),
                    images=batch['images'].half().cuda(),
                    image_sizes=batch['image_sizes'],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True)

                new_scores = []
                # print("output_scores len:", len(output.scores))
                output_scores_all = torch.stack(output.scores, dim=0)
                # print(output_scores_all.shape)
                output_scores_reshape = (batch['input_ids'].shape[0], len(output.scores), args.num_beam, output.scores[0].shape[-1])
                new_output_scores = output_scores_all.view(output_scores_reshape)

                for question, output_ids, output_scores, question_id, metainfos in zip(batch['raw_questions'], output.sequences, new_output_scores, batch['question_id'], batch['metainfos']):

                    response = tokenizer.decode(
                        output_ids, skip_special_tokens=True)
                    response = response.strip()

                    scores = torch.softmax(output_scores.squeeze(), dim=0)
                    max_value, max_index = torch.max(scores, dim=0)

                    item_scores = {
                        'yes': scores[yes_id].cpu().item(),
                        'Yes': scores[Yes_id].cpu().item(),
                        'no': scores[no_id].cpu().item(),
                        'No': scores[No_id].cpu().item()
                    }

                    if 'ds_question_id' in metainfos:
                        outputs.append({
                            'question_id': question_id,
                            'ds_question_id': metainfos['ds_question_id'],
                            'raw_question': question,
                            'answer': response,
                            'scores': item_scores,
                            'metainfos': metainfos,
                            'model_path': args.checkpoint
                        })
                    else:
                        outputs.append({
                        'question_id': question_id,
                        'raw_question': question,
                        'answer': response,
                        'scores': item_scores,
                        'metainfos': metainfos,
                        'model_path': args.checkpoint
                    })

            else:
                if args.num_beam >= 1:
                    print("use beamsearch:", args.num_beam)
                    output = model.generate(
                        inputs=batch['input_ids'].cuda(),
                        images=batch['images'].half().cuda(),
                        image_sizes=batch['image_sizes'],
                        do_sample=False,
                        num_beams=args.num_beam,
                        max_new_tokens=args.max_tokens,
                        use_cache=True,
                        return_dict_in_generate=True)
                else:
                    print("use sampling:", args.temperature)
                    output = model.generate(
                        inputs=batch['input_ids'].cuda(),
                        images=batch['images'].half().cuda(),
                        image_sizes=batch['image_sizes'],
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_tokens,
                        use_cache=True,
                        return_dict_in_generate=True)

                # print(output.scores, flush=True)
                for question, output_ids, question_id, metainfos in zip(batch['raw_questions'], output.sequences, batch['question_id'], batch['metainfos']):
                    response = tokenizer.decode(
                            output_ids, skip_special_tokens=True)
                    response = response.strip()

                    if 'ds_question_id' in metainfos:
                        outputs.append({
                            'question_id': question_id,
                            'ds_question_id': metainfos['ds_question_id'],
                            'raw_question': question,
                            'answer': response,
                            'metainfos': metainfos,
                            'model_path': args.checkpoint
                        })
                    else:
                        outputs.append({
                            'question_id': question_id,
                            'raw_question': question,
                            'answer': response,
                            'metainfos': metainfos,
                            'model_path': args.checkpoint
                        })

            cnt += 1
            if cnt == 10:
                torch.distributed.barrier()
                cnt = 0

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
    print(f'Merged outputs: {len(merged_outputs)}')
    question_ids = [x['question_id'] for x in merged_outputs]

    if torch.distributed.get_rank() == 0:
        print(f"Evaluating {args.ds_name} ...", flush=True)
        answers_file_path = args.answer_file

        with open(answers_file_path, 'w', encoding='utf-8') as f:
            json.dump(merged_outputs, f, ensure_ascii=False)

    torch.distributed.barrier()

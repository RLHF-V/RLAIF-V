import io
import os
from copy import deepcopy

import datasets
import pandas as pd
import torch
import torch.utils.data as torch_data
from PIL import Image
import tqdm
import itertools


class SampleDataset(torch_data.Dataset):
    def __init__(self, file, question_process, repeat_time=10):
        '''
        file: file that each line is a dict like {
            'image': {
                'bytes': Binary Data,
            },
            'question': question_text
        }
        '''
        super().__init__()
        self.file = file
        self.data = datasets.load_dataset(self.file, cache_dir='./cache')['train'].cast_column(
            'image',
            datasets.Image(decode=False)
        )

        # print("org data len:", len(self.data), f"\nstart={start} end={end}")
        # if end != -1 or start != 0:
        #     if end == -1:
        #         end = len(self.data)
        #     self.data = self.data.select(range(start, end))

        new_data = []
        for i in range(len(self.data)):
            new_data += [self.data[i]] * repeat_time

        self.data = new_data
        self.question_process = question_process
        self.start_idx = 0

    def __getitem__(self, index):
        item = self.data[index]
        # print(item.keys())
        if "image" in item.keys():
            img = item['image']['bytes']
            raw_img = img
            image = Image.open(io.BytesIO(img)).convert('RGB')
        elif "image_path" in item.keys():
            # print("in")
            image = Image.open(item['image_path']).convert('RGB')
            raw_img = image
        elif "image_path" in item['metainfos'].keys():
            # print("in metainfos")
            image = Image.open(item['metainfos']['image_path']).convert('RGB')
            raw_img = image

        metainfo = {key: value for key, value in item.items() if key not in ["image_id", "question", "image"]}
        raw_question = item['question']
        question_input_ids = self.question_process(raw_question)

        res = {
            'question_id': item['question_id'] if 'question_id' in item else self.start_idx + index,
            'image': image,
            'raw_image': raw_img,
            'question_input_ids': question_input_ids,
            'raw_question': raw_question,
            'metainfos': metainfo,
            'origin_dataset': self.file
        }
        res['idx'] = item['idx'] if 'idx' in item else res['question_id']
        return res

    def __len__(self):
        return len(self.data)


def sample_and_record(dataloader, model_path, model, tokenizer, answer_dir, temperature=0.7, max_tokens=512):
    outputs = []
    cnt = 0
    meta_info_field = ["raw_questions", "image_sizes", "raw_images", "question_id", "idx", "origin_dataset", "metainfos"]
    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader, f'Generating answers'):
            batch_cp = deepcopy(batch)
            for field in meta_info_field:
                if field in batch_cp:
                    del batch_cp[field]
            output = model.generate(
                **batch_cp,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                tokenizer=tokenizer,
                use_cache=True,
                return_dict_in_generate=True)

            for question, output_ids, idx, question_id, metainfos, raw_image in zip(batch['raw_questions'],
                                                                                    output.sequences,
                                                                                    batch['idx'],
                                                                                    batch['question_id'],
                                                                                    batch['metainfos'],
                                                                                    batch['raw_images']):
                response = tokenizer.decode(output_ids, skip_special_tokens=True) if not isinstance(output_ids, str) else output_ids
                response = response.strip()

                if 'ds_question_id' in metainfos:
                    outputs.append({
                        'idx': idx,
                        'question_id': question_id,
                        'ds_question_id': metainfos['ds_question_id'],
                        'question': question,
                        'chosen': response,
                        'rejected': response,
                        'image': raw_image,
                        'metainfos': metainfos,
                        'model_path': model_path
                    })
                else:
                    outputs.append({
                        'idx': idx,
                        'question_id': question_id,
                        'question': question,
                        'chosen': response,
                        'rejected': response,
                        'image': raw_image,
                        'metainfos': metainfos,
                        'model_path': model_path
                    })

            cnt += 1
            if cnt == 10:
                torch.distributed.barrier()
                cnt = 0

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
            df = pd.DataFrame(temp_data)
            output_file = os.path.join(
                answer_dir,
                f'RLAIF-V-Dataset-sampled_{idx:03}-{len(temp_data)}.parquet'
            )
            temp_file = output_file + '.tmp'
            df.to_parquet(temp_file)
            os.rename(temp_file, output_file)

    torch.distributed.barrier()

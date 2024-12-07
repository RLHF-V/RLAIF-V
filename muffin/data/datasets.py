import io
import gc
import os
import json
import random
import numpy
import torch
import base64
import pandas as pd
import os.path as op
import torch.utils.data as torch_data

from PIL import Image
from typing import List, Iterator
from muffin.data.tsv_file import TSVFile
from torch.utils.data.sampler import Sampler
from muffin.data.data_processors import register_data_processor
from muffin.eval.muffin_inference_logp import inference_logp
import datasets as hf_datasets

def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image

class RLAIFVDataset(torch_data.Dataset):
    def __init__(self, data_dir: str, reference_model=None,
                 tokenizer=None, image_token_len=None, img_processor=None, use_im_start_end=True, is_llava15=False):
        super().__init__()

        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        data_path = [file for file in os.listdir(data_dir) if file.endswith('.parquet') and 'logp' in file]
        self.data_path = data_dir

        if len(data_path) == 0:
            assert reference_model is not None, "`reference_model` is mandatory when logps do not exist."

            if not op.exists('./RLAIF-V-Dataset'):
                os.mkdir('./RLAIF-V-Dataset')
            hf_data = hf_datasets.load_dataset('openbmb/RLAIF-V-Dataset', cache_dir='./RLAIF-V-Dataset')['train'].cast_column("image", hf_datasets.Image(decode=False))

            inference_logp(reference_model, tokenizer, hf_data, self.data_path,
                            image_token_len, img_processor, use_im_start_end, is_llava15=is_llava15)

            torch.distributed.barrier()

            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))
        else:
            self.data = hf_datasets.load_dataset(data_dir)['train'].cast_column("image", hf_datasets.Image(decode=False))

        self.line_idx = list(range(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[self.line_idx[index]]
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image = bytes_to_PIL_image(sample['image']['bytes'])

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']

        return data_dict


class ChunckedRandomSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size=5000) -> None:
        self.data_source = data_source
        self.chunk_size = chunk_size

    def __iter__(self):
        n = len(self.data_source)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        print(f'Chuncked Random Sampler seed is {seed}')
        generator = torch.Generator()
        generator.manual_seed(seed)

        for st in torch.randperm(n // self.chunk_size, generator=generator).tolist():
            base = st * self.chunk_size
            for i in torch.randperm(self.chunk_size, generator=generator).tolist():
                yield base + i

        base = (n // self.chunk_size) * self.chunk_size
        for i in torch.randperm(n % self.chunk_size, generator=generator).tolist():
            yield base + i

    def __len__(self) -> int:
        return len(self.data_source)


class SingleDataSourceDataset(torch_data.Dataset):
    def __init__(self, ds_name, data_dir, tsv_filenames: List[str], intent='sft', shuffle=False) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.filenames = tsv_filenames
        self.ds_name = ds_name

        self.sizes = []
        for filename in self.filenames:
            try:
                size = int(filename[:-4].split('-')[-1])
            except:
                raise ValueError(
                    f'TSV Data File {filename} is not valid, last component separated by `-` must be the number of sample in this file')
            self.sizes.append(size)

        self.file_border_index = []
        self.prepare_border_index()

        self.files = self.filenames[:]
        self.intent = intent

        self.fetch_count = 0
        self.clear_at_n_fetch = 1000 + random.randint(100, 1000)

        self.shuffle = shuffle
        self.line_numbers = list(range(len(self)))
        if self.shuffle:
            print(f'Shuffle single dataset {ds_name}', flush=True)
            if len(self.line_numbers) >= 50_000_000:
                self.line_numbers = list(ChunckedRandomSampler(self))
            else:
                random.shuffle(self.line_numbers)

    def prepare_border_index(self):
        self.file_border_index = [0]

        temp_sum = 0
        for size in self.sizes:
            temp_sum += size
            self.file_border_index.append(temp_sum)

    def get_file_idx_and_row_idx(self, index):
        found = False
        file_idx = -1

        for border_idx, border in enumerate(self.file_border_index):
            if index < border:
                file_idx = border_idx - 1
                found = True
                break
        if not found:
            raise ValueError(
                f'Index {index} out of range for {self.size_sum} border markers')

        offset = self.file_border_index[file_idx]
        row_idx = index - offset
        return file_idx, row_idx

    def __len__(self):
        return self.file_border_index[-1]

    def __getitem__(self, index, error_count=0):
        index = self.line_numbers[index]
        self.fetch_count += 1
        if self.fetch_count >= self.clear_at_n_fetch:
            self.fetch_count = 0

            # only apply to super large datasets in fact
            if len(self.filenames) >= 50:
                self.files = self.filenames[:]
            gc.collect()

        file_idx, row_idx = self.get_file_idx_and_row_idx(index)

        try:
            data_record = self.fetch_sample(file_idx, row_idx)
            return data_record
        except Exception as e:
            print(
                f'Encounter error while reading line-{row_idx} from {self.filenames[file_idx]}')
            print(e, flush=True)
            if error_count >= 3:
                raise e
            else:
                return self.__getitem__(index + 1, error_count + 1)

    def fetch_sample(self, file_idx, row_idx):
        file = self.files[file_idx]
        if isinstance(file, str):
            self.prepare_file(file_idx)
            file = self.files[file_idx]

        assert isinstance(
            file, TSVFile), f'Expecting TSVFile but get {file} as {type(file)}'

        # tsv line as tuple
        sample = file[row_idx]
        ds_name, *values = sample

        # data dict
        sample = register_data_processor[self.ds_name](
            *values, intent=self.intent)

        if row_idx + 1 == len(file):
            del file
            # TODO: might have to update to clean memory usage?
            self.files[file_idx] = self.filenames[file_idx]

        return sample

    def prepare_file(self, idx):
        filename = self.filenames[idx]
        file = TSVFile(op.join(self.data_dir, filename))
        self.files[idx] = file


class MultiDataSourceDataset(torch_data.Dataset):
    def __init__(self, data_sources: List[SingleDataSourceDataset], data_source_weights: List[int], shuffle=False):
        super().__init__()

        self.ds_list = data_sources

        self.sum_weight = sum(data_source_weights)
        self.ds_weights = data_source_weights
        for weight in self.ds_weights:
            assert isinstance(weight, int), 'weight must be integer'

        self.offset2ds = {}
        self.offset2wt = {}
        self.offset2pd = {}
        self.prepare_offset2ds()

        ds_loops = []
        for ds, wt in zip(self.ds_list, self.ds_weights):
            ds_loop = len(ds) // wt
            ds_loops.append(ds_loop)
        max_loop = max(ds_loops)
        self.size = max_loop * self.sum_weight

        if shuffle:
            for ds in self.ds_list:
                assert ds.shuffle, f'Single dataset {ds} not shuffled, but multi-source dataset required to be shuffled'

    def prepare_offset2ds(self):
        offset = 0
        for ds, weight in zip(self.ds_list, self.ds_weights):
            pd = offset
            for _ in range(weight):
                self.offset2ds[offset] = ds
                self.offset2wt[offset] = weight
                self.offset2pd[offset] = pd
                offset += 1

    def __getitem__(self, index):
        n_loop = index // self.sum_weight
        offset = index % self.sum_weight

        ds = self.offset2ds[offset]
        ds_inner_idx = n_loop * \
                       self.offset2wt[offset] + offset - self.offset2pd[offset]
        ds_inner_idx = ds_inner_idx % len(ds)

        return ds[ds_inner_idx]

    def __len__(self):
        return self.size


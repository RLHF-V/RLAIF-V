from functools import partial

import torch.utils.data as torch_data
import json

from muffin.data.datasets import bytes_to_PIL_image
from muffin.train.train_utils import encode_multimodal_preference_sample, preprocess_v1
from omnilmm.train.train_utils import omni_preprocess


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 model_name,
                 data,
                 tokenizer,
                 image_token_len,
                 img_processor,
                 use_im_start_end=True):
        self.data = data

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end,
            'keep_image_tag': True
        }
        self.tokenizer = tokenizer

        lower_name = model_name.lower()
        if "onmi" in lower_name or ('rlaif' in lower_name and '12b' in lower_name):
            self.preprocess_func = omni_preprocess
        else:
            self.preprocess_func = partial(preprocess_v1, has_image=True)

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }
        if sample['origin_split'] is not None and sample['origin_split'] != "":
            metainfo["origin_split"] = json.loads(sample['origin_split'])
        else:
            metainfo["origin_split"] = ""

        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image = bytes_to_PIL_image(sample['image']['bytes'])
        # image = bytes_to_PIL_image(sample['image_bytes']['bytes'])

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            formated_sample, self.tokenizer, self.mm_cfg, preprocess_func=self.preprocess_func)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)

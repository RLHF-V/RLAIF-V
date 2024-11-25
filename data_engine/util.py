import torch.utils.data as torch_data
from functools import partial
from muffin.train.train_utils import SFT_collator_fn
import numpy as np
import datasets as hf_datasets
from transformers.image_processing_utils import BatchFeature

from builder.builder import load_pretrained_model
from muffin.eval.muffin_inference_logp import (InferenceSampler, concate_pad)
from dataset import PreferenceInferenceDataset

import torch


def preference_collator_fn(instances, pad_token_id, is_omni=False):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    if not is_omni:
        if isinstance(win_batch['images'][0], BatchFeature):
            win_images = torch.stack([torch.tensor(img.pixel_values[0]) for img in win_batch['images']])
        elif isinstance(win_batch['images'][0], np.ndarray):
            win_images = torch.stack([torch.tensor(img) for img in win_batch['images']])
        else:
            win_images = win_batch['images']

    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'] if is_omni else win_images,
    )
    return batch


def load_model_and_dataloader(model_path, model_name, dataset_path):
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device_map={"": 'cuda'})
    image_token_len = 0
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "num_query"):
        image_token_len = model.model.config.num_query

    model = model.to(dtype=torch.bfloat16, device='cuda')
    hf_data = hf_datasets.load_dataset(dataset_path, cache_dir='./cache')['train'].cast_column("image",
                                                                                               hf_datasets.Image(
                                                                                                   decode=False))
    dataset = PreferenceInferenceDataset(model_name=model_name,
                                         tokenizer=tokenizer,
                                         data=hf_data,
                                         image_token_len=image_token_len,
                                         img_processor=image_processor,
                                         use_im_start_end=False)
    collate_fn = partial(
        preference_collator_fn,
        pad_token_id=tokenizer.pad_token_id,
        is_omni=("omni" in model_name.lower()) or (
                "rlaif" in model_name.lower() and "12b" in model_path.lower()))  # judge if the model follow omni structure
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False, sampler=InferenceSampler(len(dataset)))
    return model, dataset, dataloader

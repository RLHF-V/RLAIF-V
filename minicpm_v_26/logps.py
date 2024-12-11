import datasets
import math
import re
from dataclasses import dataclass
from typing import Dict, Callable, Sequence

import numpy as np
import transformers
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import logging

import os
import json
import copy
import torch
import pickle

import tqdm
import itertools
from PIL import Image

from transformers import AutoTokenizer
import torch.utils.data as torch_data
from torch.utils.data import Dataset

import builder.builder
from muffin.data.data_processors import b64_to_PIL_image
from muffin.eval.muffin_inference_logp import bytes_to_PIL_image
from muffin.gen_data_util import InferenceSampler

logger = logging.getLogger(__name__)


class PreferenceInferenceDataset(Dataset):
    def __init__(self,
                 file,
                 tokenizer=None,
                 transform=None,
                 slice_config=None,
                 batch_vision=True,
                 max_length=2048,
                 ):

        self.data = datasets.load_dataset(file, cache_dir='./cache')['train'].cast_column(
            'image',
            datasets.Image(decode=False)
        )

        self.tokenizer = tokenizer
        self.transform = transform
        self.slice_config = slice_config
        self.batch_vision = batch_vision
        self.max_length = max_length

    def preprocess_input(self, image_list, msgs):
        model_inputs = preprocess(
            images_dict=image_list,
            conversations=msgs,
            tokenizer=self.tokenizer,
            transform=self.transform,
            slice_config=self.slice_config,
            batch_vision=self.batch_vision,
            max_length=self.max_length
        )

        model_inputs = dict(
            input_ids=model_inputs["input_ids"],
            position_ids=model_inputs["position_ids"],
            labels=model_inputs["target"],
            attention_mask=torch.ones_like(model_inputs["input_ids"], dtype=torch.bool),
            pixel_values=model_inputs.get("pixel_values", None),
            tgt_sizes=model_inputs.get("tgt_sizes", None),
            image_bound=model_inputs["image_bound"],
        )

        return model_inputs

    def prepare_inputs(self, index):
        try:
            sample = copy.deepcopy(self.data[index])
        except:
            sample = copy.deepcopy(self.data.iloc[index])

        question = {'role': 'user', 'content': f"<image>\n{sample['question']}"}
        chosen = {'role': 'assistant', 'content': sample['chosen']}
        rejected = {'role': 'assistant', 'content': sample['rejected']}

        if 'image' in sample.keys():
            images_dict = {"<image>": bytes_to_PIL_image(sample['image']['bytes'])}
        elif 'image_path' in sample.keys() and isinstance(sample['image_path'], str):
            images_dict = {"<image>": Image.open(os.path.join(self.img_dir, sample['image_path'])).convert("RGB")}

        formated_sample = {
            'image': images_dict,
            "chosen": [question, chosen],
            "rejected": [question, rejected],
            "idx": sample['idx'],
        }

        return formated_sample

    def __getitem__(self, index):
        formated_sample = self.prepare_inputs(index)

        # return formated_sample

        sample = {
            "chosen": self.preprocess_input(formated_sample['image'], formated_sample['chosen']),
            "rejected": self.preprocess_input(formated_sample['image'], formated_sample['rejected'])
        }

        return sample

    def __len__(self):
        return len(self.data)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False,
                    return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    loss_mask = labels != -100
    labels[labels == -100] = 0  # dummy token
    per_token_logps = torch.gather(logits.softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    if return_per_token_logp:
        return per_token_logps

    return log_prob, average_log_prob


def save_logp_pkl(data, cache_file, logps, overwrite_logps=False):
    out_data = []

    for index in range(len(logps)):
        try:
            line = data[index]
        except:
            line = data.iloc[index]
        logp_data = {}
        logp_data['logps'] = logps[index]

        new_line = copy.deepcopy(line)

        if 'logps' in new_line.keys():
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            new_line['logps'] = json.dumps(logp_data)

        else:
            assert (('question' in list(new_line.keys()))
                    and ('chosen' in list(new_line.keys()))
                    and ('rejected' in list(new_line.keys()))), \
                f'Undefined data structure, expecting [Q, Win, Rej] in keys, got {new_line.keys()}'
            new_line['logps'] = json.dumps(logp_data)

        out_data.append(new_line)

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(out_data, f)


class PreferenceModel:
    def __init__(self, model, tokenizer, max_length=2048) -> None:
        self.model = model

        self.model = self.model.to(device='cuda')
        self.tokenizer = tokenizer

        self.model.eval()
        self.config = self.model.config

        self.max_length = max_length

        if hasattr(self.model.config, "slice_config"):
            self.model.config.slice_config.max_slice_nums = 2
            slice_config = self.model.config.slice_config.to_dict()
        else:
            self.model.config.max_slice_nums = 2
            slice_config = self.model.config.to_dict()
        self.slice_config = slice_config
        # print("self.slice_config:", self.slice_config)

    def inference_logp(self, sample, ans_key):
        assert len(sample[ans_key]) == 1, f'len(sample[ans_key]) = {len(sample[ans_key])}'
        model_inputs = sample[ans_key][0]

        for key in model_inputs:
            if isinstance(model_inputs[key], list):
                model_inputs[key] = [model_inputs[key][i].to(self.model.device) for i in range(len(model_inputs[key]))]
            else:
                model_inputs[key] = model_inputs[key].to(self.model.device)

        model_inputs = data_collator([model_inputs], max_length=self.max_length)

        with torch.inference_mode():
            output = self.model(
                model_inputs
            )

            per_token_logp, log_prob, average_log_prob = get_batch_logps(
                output.logits, model_inputs['labels'], self.tokenizer, return_all=True)

            assert per_token_logp.size(1) >= model_inputs['input_ids'].size(1) - 1
            per_token_logp = per_token_logp.tolist()
            log_prob = log_prob.tolist()
            average_log_prob = average_log_prob.tolist()

        return per_token_logp, log_prob, average_log_prob


def get_multimodal_sample_logps(model, dataloader):
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        idx = 0
        for batch in tqdm.tqdm(dataloader):
            for key in ['chosen', 'rejected']:
                per_token_logp, log_prob, average_log_prob = model.inference_logp(
                    sample=batch,
                    ans_key=key
                )

                if key == 'chosen':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp

            idx += 1

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list


def colloator_fn(data_list):
    data = {}
    for key in data_list[0]:
        data[key] = [x[key] for x in data_list]

    return data


def get_dataset_inference_logp(model_name, model_path, data_path, transform=None, slice_config=None,
                               batch_vision=True, max_length=2048):
    tokenizer, model, _, _ = builder.builder.load_pretrained_model(model_path, None, model_name)
    model = PreferenceModel(model, tokenizer, max_length=max_length)
    dataset = PreferenceInferenceDataset(data_path,
                                         tokenizer=tokenizer,
                                         transform=transform if transform is not None else build_transform(),
                                         slice_config=slice_config if slice_config is not None else model.slice_config,
                                         batch_vision=batch_vision,
                                         max_length=max_length)

    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=colloator_fn,
                                       num_workers=5, shuffle=False,
                                       sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model,
                                          dataloader)  # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]

    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list,
                     rej_per_token_logp_list))
    torch.distributed.barrier()

    del model
    return logps, dataset.data


@dataclass
class PreferenceDatasetDataCollator(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    preference_collator_fn: Callable

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.preference_collator_fn(instances)

        rej_instances, win_instances = list(zip(*instances))

        batch['beta'] = self.beta
        batch['ref_win_logp'] = torch.as_tensor(
            [x['ref_win_logp'] for x in win_instances])
        batch['ref_rej_logp'] = torch.as_tensor(
            [x['ref_rej_logp'] for x in rej_instances])
        batch['ref_win_avg_logp'] = torch.as_tensor(
            [x['ref_win_avg_logp'] for x in win_instances])
        batch['ref_rej_avg_logp'] = torch.as_tensor(
            [x['ref_rej_avg_logp'] for x in rej_instances])

        # print("datacollator_dpodataset: batch['ref_win_logp']:", batch['ref_win_logp'].dtype, batch['ref_win_logp'].item())

        ref_win_per_token_logp = [torch.as_tensor(
            x['ref_win_per_token_logp']) for x in win_instances]
        ref_rej_per_token_logp = [torch.as_tensor(
            x['ref_rej_per_token_logp']) for x in rej_instances]

        batch['ref_win_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_win_per_token_logp, batch_first=True, padding_value=0)
        batch['ref_rej_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_rej_per_token_logp, batch_first=True, padding_value=0)

        win_input_ids = batch['win_input_ids']
        rej_input_ids = batch['rej_input_ids']

        assert batch['ref_win_per_token_logp'].size(1) >= win_input_ids.size(
            1) - 1, f"{batch['ref_win_per_token_logp'].size(1)} >= {win_input_ids.size(1) - 1}, {self.tokenizer.batch_decode(win_input_ids)}"
        assert batch['ref_rej_per_token_logp'].size(1) >= rej_input_ids.size(
            1) - 1, f"{batch['ref_rej_per_token_logp'].size(1)} >= {rej_input_ids.size(1) - 1}"

        # length of logp is one-token shorter since the last token's output is not used
        batch['ref_win_per_token_logp'] = batch['ref_win_per_token_logp'][:,
                                          :win_input_ids.size(1) - 1]
        batch['ref_rej_per_token_logp'] = batch['ref_rej_per_token_logp'][:,
                                          :rej_input_ids.size(1) - 1]
        for ins in win_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in rej_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        return batch


def data_collator(examples, padding_value=0, max_length=2048):
    def trim_and_pad(seq, batch_first, padding_value):
        return pad_sequence([s[:max_length] for s in seq], batch_first=batch_first, padding_value=padding_value)

    input_ids = trim_and_pad(
        [example["input_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    position_ids = trim_and_pad(
        [example["position_ids"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    targets = trim_and_pad(
        [example["labels"] for example in examples],
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = trim_and_pad(
        [example["attention_mask"] for example in examples],
        batch_first=True,
        padding_value=padding_value,
    )
    pixel_values = [example["pixel_values"] for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    tgt_sizes = [example["tgt_sizes"] for example in examples]
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
    }
    ### <===


def preference_collator_fn(instances, pad_token_id=0, max_length=2048):
    def concate_pad(tensorA, tensorB, padding_value):
        out = torch.nn.utils.rnn.pad_sequence(
            list(tensorA) + list(tensorB),
            batch_first=True,
            padding_value=padding_value)
        return out

    rej_instances, win_instances = list(zip(*instances))
    rej_batch = data_collator(rej_instances, pad_token_id, max_length=max_length)
    win_batch = data_collator(win_instances, pad_token_id, max_length=max_length)

    # 把正例和反例丢到一个batch中
    concatenated_input_ids = concate_pad(
        win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(
        win_batch['labels'], rej_batch['labels'], -100)
    concatenated_position_ids = concate_pad(
        win_batch['position_ids'], rej_batch['position_ids'], pad_token_id)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    batch = dict(
        concatenated_input_ids=concatenated_input_ids,  # 这个字段似乎在data_collator中被丢弃了
        concatenated_labels=concatenated_labels,  # 这个字段似乎在data_collator中被丢弃了
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],

        concatenated_position_ids=concatenated_position_ids,  # 这个字段似乎在data_collator中被丢弃了
        concatenated_attention_mask=concatenated_attention_mask,  # 这个字段似乎在data_collator中被丢弃了

        images=win_batch['pixel_values'] + win_batch['pixel_values'],
        image_bound=win_batch['image_bound'] + win_batch['image_bound'],
        tgt_sizes=win_batch['tgt_sizes'] + rej_batch['tgt_sizes']
    )

    # print("batch['images']:", len(batch['images']), len(batch['images'][0]))
    # print("batch['image_bounds']:", batch['image_bounds'])
    # print("batch['tgt_sizes']:", batch['tgt_sizes'])

    return batch


def conversation_to_ids(conversation, tokenizer, max_length=2048):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """

    input_ids, context, raw_msg = llm_conversation_to_ids(
        conversation, tokenizer
    )

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))
    if input_ids.shape[-1] > max_length:
        ids = ids[:max_length]
        context = context[:max_length]
        logger.warning(
            f"The input length ({input_ids.shape[-1]}) exceeds the model's maximum length ({max_length}), so it has been truncated")

    if torch.all(context):
        logger.error("No tokens available to compute loss.")
        raise Exception("No tokens available to compute loss.")

    target = build_conversation_ids_target(ids, context, tokenizer)
    image_bound = build_conversation_ids_image_bound(ids, tokenizer)

    position_ids = torch.arange(ids.size(0)).long()
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }


def llm_conversation_to_ids(conversation, tokenizer):
    raw_msg = ""
    chat = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        if not isinstance(message, str):
            message = message[0]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "user"
        else:
            prefix = "assistant"
        chat.append({"role": prefix, "content": message})
        raw_msg += prefix + message
    assert set([i['role'] for i in chat]) & set(['assistant'])

    ret = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids[:-1])

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx - 1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg


def build_conversation_ids_target(ids, context, tokenizer):
    # build target
    target = torch.full_like(ids, -100, dtype=torch.int64)  # dtype=torch.int32

    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id
    return target


def build_conversation_ids_image_bound(ids, tokenizer):
    # build image bound
    start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
    end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
    image_start_tokens = torch.where(start_cond)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(end_cond)[0]

    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")

    if len(image_start_tokens) > 0:
        image_bound = torch.hstack(
            [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
        )
    else:
        image_bound = []
    return image_bound


def preprocess(
        images_dict,
        conversations,
        tokenizer,
        transform,
        query_nums=64,
        slice_config=None,
        patch_size=14,
        batch_vision=False,
        max_length=2048,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )

    use_image_id = True
    image_placeholder_dict, images, image_placeholder = preprocess_image(images_dict, tokenizer, transform, query_nums,
                                                                         slice_config, default_image_placeholder,
                                                                         use_image_id)

    if len(images_dict) == 1 and "<image>" in images_dict:
        if "<image>" in conversations[0]["content"]:
            conversations[0]["content"] = conversations[0]["content"].replace(
                "<image>", image_placeholder
            )
        else:
            conversations[0]["content"] = (
                    image_placeholder + "\n" + conversations[0]["content"]
            )
        input_dict = conversation_to_ids(conversations, tokenizer, max_length)
    else:
        # if there are multi conservations, we need to take image name into consideration
        # here it uses different <image_\d+> to distinguish different images
        pattern = r'<image_\d+>'
        new_conversations = []
        for conversation in conversations:
            content = conversation['content']
            # >>> content = "Hello <image_1> world <image_2>!"
            # >>> re.split(f'({pattern})', content)
            # ['Hello ', '<image_1>', ' world ', '<image_2>', '!']
            parts = re.split(f'({pattern})', content)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                if re.match(pattern, part):
                    if part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part]
                    else:
                        raise Exception(f"not found {part} in image dict")
            conversation['content'] = '\n'.join(parts)
            new_conversations.append(conversation)
        conversations = new_conversations

        input_dict = conversation_to_ids(conversations, tokenizer, max_length)

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def preprocess_image(images_dict, tokenizer, transform, query_nums, slice_config, default_image_placeholder,
                     use_image_id):
    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0
    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder
            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])
                if use_image_id:
                    image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                    image_id_cnt += 1
                image_placeholder += get_grid_placeholder(
                    tokenizer, best_grid, query_nums)
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder

    images = [transform(i) for i in images]
    return image_placeholder_dict, images, image_placeholder


def slice_image(
        image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
            (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
        original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
            tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))

    slice_placeholder = '\n'.join(slices)
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
    )

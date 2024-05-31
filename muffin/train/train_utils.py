import os
import gc
import copy
import time
import json

import torch
import warnings
import tokenizers
import transformers

import numpy as np

from typing import Dict, Optional, Sequence
from muffin import conversation as conversation_lib
from packaging import version


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
IMAGE_TOKEN_INDEX = -200 # from llava 1.5, used to determin image in forward function
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def SFT_collator_fn(instances, pad_token_id):
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(pad_token_id),
    )

    images = [instance['image']
              for instance in instances if 'image' in instance]
    if len(images) > 0:
        # possibly multi-image for each sample
        if len(images[0].shape) == 4:
            batch['images'] = images
        elif all(x is not None and x.shape == images[0].shape for x in images):
            import numpy
            if isinstance(images[0], numpy.ndarray):
                images = [torch.from_numpy(x) for x in images]
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
    else:
        batch['images'] = []

    # for minicpm
    if 'context_ids' in instances[0]:
        image_bounds, context_ids, position_ids = \
            tuple([instance[key] for instance in instances]
                  for key in ("image_bounds", "context_ids", "position_ids"))
        batch['image_bounds'] = image_bounds
        batch['context_ids'] = torch.nn.utils.rnn.pad_sequence(context_ids,
                                             batch_first=True,
                                             padding_value=0)
    return batch


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def expand_image_token(source, multimodal_cfg) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = multimodal_cfg['image_token_len']
    if not is_multimodal or multimodal_cfg.get('keep_image_tag', False):
        return source

    for sentence in source:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        if multimodal_cfg['use_im_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        sentence["value"] = sentence["value"].replace(
            DEFAULT_IMAGE_TOKEN, replace_token)

    return source

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg, preprocess_func=None):
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        win_conv = expand_image_token(win_conv, multimodal_cfg)
        rej_conv = expand_image_token(rej_conv, multimodal_cfg)

    if preprocess_func is None:
        rej_data_dict = preprocess([rej_conv], tokenizer)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], tokenizer)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0])
    else:
        rej_data_dict = preprocess_func([rej_conv], tokenizer)
        win_data_dict = preprocess_func([win_conv], tokenizer)

        if 'context_ids' in rej_data_dict:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0],
                                image_bounds=rej_data_dict['image_bounds'][0],
                                context_ids=rej_data_dict['context_ids'][0],
                                position_ids=rej_data_dict['position_ids'][0]
                                )
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0],
                                image_bounds=win_data_dict['image_bounds'][0],
                                context_ids=win_data_dict['context_ids'][0],
                                position_ids=win_data_dict['position_ids'][0]
                                )
        else:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0])
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0])

    # print(f'rej dict: {rej_data_dict}', flush=True)
    # print('inputs:', tokenizer.decode([(x if x != -200 else 0) for x in rej_data_dict['input_ids'].tolist()]), flush=True)
    # print('labels:', tokenizer.decode([(x if x != -100 else 0) for x in rej_data_dict['labels'].tolist()]), flush=True)

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])

    if 'ref_win_logp' in source:
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
    return rej_data_dict, win_data_dict

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            print("Skip the first one if it is not from human")
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


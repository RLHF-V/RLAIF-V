import os
import json
import tqdm
import torch
import base64
import torch.utils.data as torch_data
from muffin.train.train_utils import SFT_collator_fn
from typing import List
from functools import partial


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out


def get_batch_logps_minicpm(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, :-1].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    assert per_token_logps.shape == labels.shape, f"per_token_logps.shape={per_token_logps.shape}, labels.shape={labels.shape}"
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'{logits.shape}, {labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob


def preference_collator_fn(instances, pad_token_id, is_kto=False):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(
        win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(
        win_batch['labels'], rej_batch['labels'], -100)
    # concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    if is_kto:
        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            win_input_ids=win_batch['input_ids'],
            rej_input_ids=rej_batch['input_ids'],
            win_labels=win_batch['labels'],
            rej_labels=rej_batch['labels'],
            win_images=win_batch['images'],
            rej_images=rej_batch['images']
        )
    elif 'context_ids' in win_batch:
        concatenated_context_ids = concate_pad(
            win_batch['context_ids'], rej_batch['context_ids'], 1)

        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            win_input_ids=win_batch['input_ids'],
            rej_input_ids=rej_batch['input_ids'],
            win_labels=win_batch['labels'],
            rej_labels=rej_batch['labels'],
            win_context_ids=win_batch['context_ids'],
            rej_context_ids=rej_batch['context_ids'],

            images=win_batch['images'] + win_batch['images'],
            context_ids=concatenated_context_ids,
            image_bounds=win_batch['image_bounds'] + win_batch['image_bounds'],
        )
    else:
        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            # concatenated_attention_mask=concatenated_attention_mask,
            win_input_ids=win_batch['input_ids'],
            rej_input_ids=rej_batch['input_ids'],
            win_labels=win_batch['labels'],
            rej_labels=rej_batch['labels'],
            # win_attention_mask=win_batch['attention_mask'],
            # rej_attention_mask=rej_batch['attention_mask'],
            images=win_batch['images'],
        )
    return batch

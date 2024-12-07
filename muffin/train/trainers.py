from torch import nn
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
import os
import math
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from utils.utils import is_main_process

from muffin.eval.muffin_inference_logp import get_batch_logps, get_batch_logps_minicpm


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

class ZephyrTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        # Build the sampler.
        return RandomSampler(self.train_dataset)
        # return SequentialSampler(self.train_dataset)

        # if self.args.group_by_length:
        #     assert NotImplementedError
        # else:
        #     if len(self.train_dataset) >= 50_000_000:
        #         return ChunckedRandomSampler(self.train_dataset)
        #     else:
        #         # print(f'Data set size is :{len(self.train_dataset)}', flush=True)
        #         # return SequentialSampler(self.train_dataset)

        #         print(f'Shuffle Data set size is :{len(self.train_dataset)}', flush=True)
        #         return RandomSampler(self.train_dataset)

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps_minicpm if is_minicpm else get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    # print(f'compute wlogp {labels.shape} {loss_mask.shape}, {token_weight.shape}, {per_token_logp.shape}')
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


def collect_preference_metrics(metrics, task,
                               chosen_rewards, rejected_rewards,
                               policy_rej_logp, policy_win_logp,
                               ref_rej_logp, ref_win_logp, reward_accuracies,
                               preprocess_func,
                               ):
    t = task
    metrics = {}
    metrics[f'rewards_{t}/chosen'] = preprocess_func(chosen_rewards)
    metrics[f'rewards_{t}/rejected'] = preprocess_func(rejected_rewards)
    metrics[f'logps_{t}/rejected'] = preprocess_func(policy_rej_logp)
    metrics[f'logps_{t}/chosen'] = preprocess_func(policy_win_logp)
    metrics[f'logps_{t}/ref_rejected'] = preprocess_func(ref_rej_logp)
    metrics[f'logps_{t}/ref_chosen'] = preprocess_func(ref_win_logp)
    metrics[f'rewards_{t}/accuracies'] = preprocess_func(
        reward_accuracies)
    metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - \
        metrics[f'rewards_{t}/rejected']
    return metrics


def get_beta_and_logps(data_dict, model, args, is_minicpm=False, is_llava15=False):
    win_input_ids = data_dict.pop('win_input_ids')
    rej_input_ids = data_dict.pop('rej_input_ids')

    win_labels = data_dict.pop('win_labels')
    rej_labels = data_dict.pop('rej_labels')

    win_attention_mask = data_dict.pop('win_attention_mask')
    rej_attention_mask = data_dict.pop('rej_attention_mask')

    ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
    ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
    ref_win_logp = data_dict.pop('ref_win_logp')
    ref_rej_logp = data_dict.pop('ref_rej_logp')
    ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
    ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')
    if args.dpo_use_average:
        ref_win_logp = ref_win_avg_logp
        ref_rej_logp = ref_rej_avg_logp

    beta = data_dict.pop('beta')
    if args.task == 'DPO':
        images = data_dict.pop('images')
        if is_minicpm:
            # print(data_dict.keys())
            data_dict.pop('win_context_ids')
            data_dict.pop('rej_context_ids')
            concatenated_images = images
        else:
            concatenated_images = torch.cat([images, images], dim=0)
    elif args.task == 'KTO':
        win_images = data_dict.pop('win_images')
        rej_images = data_dict.pop('rej_images')
        concatenated_images = torch.cat([win_images, rej_images], dim=0)

    concatenated_input_ids = data_dict.pop('concatenated_input_ids')
    concatenated_labels = data_dict.pop('concatenated_labels')
    concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
    concatenated_attention_mask = None

    win_token_weight = data_dict.pop('win_token_weight')
    rej_token_weight = data_dict.pop('rej_token_weight')
    concatenated_token_weight = data_dict.pop('concatenated_token_weight')

    if is_llava15:
        (
            _,
            _,
            _,
            _,
            concatenated_inputs_embeds,
            concatenated_labels
        ) = model.prepare_inputs_labels_for_multimodal(
            input_ids=concatenated_input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=concatenated_labels,
            images=concatenated_images,
        )
        output = model.forward(
            inputs_embeds=concatenated_inputs_embeds,
            labels=None,
            **data_dict,
        )
        log_prob, average_log_prob = get_batch_logps(
            output.logits, concatenated_labels, return_per_token_logp=False)
        if args.dpo_use_average:
            concatenated_logp = average_log_prob
        else:
            concatenated_logp =log_prob
    else:
        concatenated_logp = forward_DPO(model,
                                        concatenated_input_ids,
                                        concatenated_labels,
                                        concatenated_attention_mask,
                                        concatenated_images,
                                        token_weighted=args.dpo_token_weighted,
                                        dpo_use_average=args.dpo_use_average,
                                        is_minicpm=is_minicpm,
                                        **data_dict)
    win_size = win_input_ids.shape[0]
    rej_size = rej_input_ids.shape[0]
    assert win_size == rej_size

    if args.dpo_token_weighted:
        if is_llava15:
            raise NotImplementedError
        # print(f'compute_loss win {win_input_ids.shape} {win_labels.shape} {ref_win_per_token_logp.shape} {win_token_weight.shape}', flush=True)
        # print(f'compute_loss rej {rej_input_ids.shape} {rej_labels.shape} {ref_rej_per_token_logp.shape} {rej_token_weight.shape}', flush=True)
        # print(f'compute_loss cat {concatenated_input_ids.shape} {concatenated_labels.shape} {concatenated_logp.shape} {concatenated_token_weight.shape}', flush=True)

        # for i in range(len(ref_win_per_token_logp)):
        #     print(f'compuate loss {i} win_input_ids={win_input_ids[i]}\nwin_labels={win_labels[i]}\nwin_per_token_logp={ref_win_per_token_logp[i]}\nwin_token_weight={win_token_weight[i]}\n', flush=True)
        #     print(f'compuate loss {i} rej_input_ids={rej_input_ids[i]}\nrej_labels={rej_labels[i]}\nrej_per_token_logp={ref_rej_per_token_logp[i]}\nrej_token_weight={rej_token_weight[i]}\n', flush=True)
        ref_win_logp = compute_weighted_logp(
            ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
        ref_rej_logp = compute_weighted_logp(
            ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
        concatenated_logp = compute_weighted_logp(
            concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

        if torch.any(torch.isnan(ref_win_logp)):
            print(f'ref_win_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(ref_rej_logp)):
            print(f'ref_rej_logp fail', flush=True)
            exit()
        if torch.any(torch.isnan(concatenated_logp)):
            print(f'concatenated_logp fail', flush=True)
            exit()

    policy_win_logp, policy_rej_logp = concatenated_logp.split(
        [win_size, rej_size])
    return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta



class LLaVA15DPOTrainer(ZephyrTrainer):

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, beta = get_beta_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp,
                                                            policy_rej_logp,
                                                            ref_win_logp,
                                                            ref_rej_logp,
                                                            beta=beta)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        loss = DPO_weight * losses.mean() - SFT_weight * policy_win_logp.mean()

        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, reward_accuracies,
                                             gather_and_do_mean)
        self.log(metrics)

        return loss

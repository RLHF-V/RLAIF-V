# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
import logging
import pathlib
import getpass
import transformers

from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from utils.utils import is_main_process, get_rank
from utils.diff_lib import get_diff_ids, color_print_diff_pair, split_into_words
from muffin.eval.muffin_inference_logp import preference_collator_fn, concate_pad

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    mod_token_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)

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
        win_labels = batch['win_labels']
        rej_labels = batch['rej_labels']
        assert batch['ref_win_per_token_logp'].size(1) >= win_input_ids.size(
            1) - 1, f"{batch['ref_win_per_token_logp'].size(1)} >= {win_input_ids.size(1) - 1}"
        assert batch['ref_rej_per_token_logp'].size(1) >= rej_input_ids.size(
            1) - 1, f"{batch['ref_rej_per_token_logp'].size(1)} >= {rej_input_ids.size(1) - 1}"

        # length of logp is one-token shorter since the last token's output is not used
        batch['ref_win_per_token_logp'] = batch['ref_win_per_token_logp'][:,
                                                                          :win_input_ids.size(1) - 1]
        batch['ref_rej_per_token_logp'] = batch['ref_rej_per_token_logp'][:,
                                                                          :rej_input_ids.size(1) - 1]

        win_token_weight = torch.ones_like(batch['ref_win_per_token_logp'])
        rej_token_weight = torch.ones_like(batch['ref_rej_per_token_logp'])

        for idx, (w, r, wl, rl, wlogp, rlogp) in enumerate(zip(win_input_ids, rej_input_ids, win_labels, rej_labels, ref_win_per_token_logp, ref_rej_per_token_logp)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(
                valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            r_mod_tokens = valid_r[r_mod]
            w_mod_tokens = valid_w[w_mod]
            win_token_weight[idx][w_mod] = self.mod_token_weight
            rej_token_weight[idx][r_mod] = self.mod_token_weight

        batch['win_token_weight'] = win_token_weight
        batch['rej_token_weight'] = rej_token_weight
        batch['concatenated_token_weight'] = concate_pad(
            win_token_weight, rej_token_weight, 0)

        for ins in win_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in rej_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        if torch.any(torch.isnan(batch['win_token_weight'])):
            print(f'win_token_weight fail', flush=True)
            exit()
        if torch.any(torch.isnan(batch['rej_token_weight'])):
            print(f'rej_token_weight fail', flush=True)
            exit()
        return batch

import io
import os
import json
import tqdm
import copy
import torch
import itertools
import pandas as pd
import torch.utils.data as torch_data
import PIL.Image as PIL_image
from functools import partial
from muffin.train.train_utils import encode_multimodal_preference_sample, SFT_collator_fn, preprocess_v1


def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = PIL_image.open(img_io).convert('RGB')
    return image

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


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print("==>", labels)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
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

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": json.loads(sample['origin_split']),
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image = bytes_to_PIL_image(sample['image']['bytes'])

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        preprocess_func= partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            formated_sample, self.tokenizer, self.mm_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)


def pretty_print(data_dict, tokenizer):
    input_ids = data_dict['input_ids']
    input_str = tokenizer.decode(input_ids)
    print(f'input_ids.shape={input_ids.shape}\ninput_str is {input_str}')

    label_ids = data_dict['labels']
    print(f'label_ids.shape={input_ids.shape}')
    for i, o in zip(input_ids, label_ids):
        i_tok = tokenizer.convert_ids_to_tokens(i.item())
        o_tok = tokenizer.convert_ids_to_tokens(o.item()) if o.item() != -100 else '[SKIP]'
        print(f'{i_tok:10s} => {o_tok:10s}')


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def preference_collator_fn(instances, pad_token_id):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

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
        images=win_batch['images'],
    )
    return batch




def get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=False):
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        idx=0
        for batch in tqdm.tqdm(dataloader):
            for key in ['win', 'rej']:
                input_ids = batch[f'{key}_input_ids'].cuda()
                # tokens = tokenizer.batch_decode(copy.deepcopy(input_ids))
                # print(tokens)
                labels = batch[f'{key}_labels'].cuda()
                attention_mask = batch[f'{key}_attention_mask'].cuda()

                if is_llava15:
                    # print("is llava15")
                    (
                        _,
                        _,
                        _,
                        _,
                        inputs_embeds,
                        labels
                    ) = model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=labels,
                        images=batch['images'].to(dtype=torch.bfloat16, device='cuda'),
                    )
                    output = model.forward(
                        inputs_embeds=inputs_embeds,
                        labels=None,
                    )
                else:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        images=batch['images'].to(dtype=torch.bfloat16, device='cuda'),
                    )
                per_token_logp, log_prob, average_log_prob = get_batch_logps(output.logits, labels, return_all=True)

                # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
                log_prob = log_prob.tolist()
                average_log_prob = average_log_prob.tolist()

                if key == 'win':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp
            # print(f'{key} logits in {output.logits.shape}, logp in {log_prob.shape} avg_logp in {average_log_prob.shape}', flush=True)

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list


def write_logp_to_preference_parquet(origin_data, cache_file, logps, overwrite_logps=False):
    out_data = []

    for index in range(len(logps)):
        line = origin_data[index]
        logp_data = {}
        logp_data['logps']=logps[index]

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

    if torch.distributed.get_rank() == 0:
        step = 5000
        for idx, start in enumerate(range(0, len(out_data), step)):
            temp_data = out_data[start: min(start+step, len(out_data))]
            df = pd.DataFrame(temp_data)
            df.to_parquet(os.path.join(cache_file, f'RLAIF-V-Dataset-withlogp_{idx:03}-{len(temp_data)}.parquet'))

    torch.distributed.barrier()

def inference_logp(model, tokenizer, hf_data, cache_file, image_token_len, img_processor, use_im_start_end, is_llava15=False):
    model = model.to(dtype=torch.bfloat16, device='cuda')
    dataset = PreferenceInferenceDataset(tokenizer=tokenizer,
                                    data = hf_data,
                                    image_token_len=image_token_len,
                                    img_processor=img_processor,
                                    use_im_start_end=use_im_start_end)
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False, sampler=InferenceSampler(len(dataset)))

    outputs = get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=is_llava15) # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]


    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    write_logp_to_preference_parquet(dataset.data, cache_file, logps, overwrite_logps=False)

    torch.distributed.barrier()

    del model

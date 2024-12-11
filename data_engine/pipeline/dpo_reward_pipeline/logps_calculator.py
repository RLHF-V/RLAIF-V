import itertools
import argparse
from functools import partial

import datasets
import numpy as np
from transformers import BatchFeature

from builder.builder import load_pretrained_model
from muffin.eval.muffin_inference_logp import (write_logp_to_preference_parquet, get_multimodal_sample_logps,
                                               concate_pad)
from muffin.gen_data_util import InferenceSampler
from muffin.train.train_utils import SFT_collator_fn
from data_engine.util import *
from data_engine.pipeline.dpo_reward_pipeline.dataset import PreferenceInferenceDataset
import minicpm_v_26.logps

import torch
import torch.distributed as dist
import torch.utils.data as torch_data


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


def inference_logp(
        model_name,
        model_path,
        dataset_path,
        output_dir):
    """
    Args:
        model_name:  e.g. llava-v1.5-7B, OmniLMM-12B, RLAIF-V-12B
        model_path: path to your model
        dataset_path: path to dataset(should follow RLAIF-V-Dataset format)
        output_dir: path to outputfile(logps)

    Returns:

    """
    if judge_is_minicpmv26(model_name):
        logps, data = minicpm_v_26.logps.get_dataset_inference_logp(model_name, model_path, dataset_path)
        _ = write_logp_to_preference_parquet(data, output_dir, logps, overwrite_logps=True)

        torch.distributed.barrier()
        return

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name,
                                                                           device_map={"": 'cuda'})
    image_token_len = 0
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "num_query"):
        image_token_len = model.model.config.num_query

    model = model.to(dtype=torch.bfloat16, device='cuda')
    hf_data = datasets.load_dataset(dataset_path, cache_dir='./cache')['train'].cast_column("image",
                                                                                            datasets.Image(
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

    outputs = get_multimodal_sample_logps(
        # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list
        model,
        dataloader,
        model_name=model_name,
        is_llava15=None)

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]

    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list,
                     rej_per_token_logp_list))

    _ = write_logp_to_preference_parquet(dataset.data, output_dir, logps, overwrite_logps=True)

    torch.distributed.barrier()


def main(
        reward_model_name: str,
        reward_model_path: str,
        instruct_model_name: str,
        instruct_model_path: str,
        dataset_path: str,
        reward_output_dir: str,
        instruct_output_dir: str):
    inference_logp(instruct_model_name, instruct_model_path, dataset_path, instruct_output_dir)
    inference_logp(reward_model_name, reward_model_path, dataset_path, reward_output_dir)

    return {
        "reward_output_dir": reward_output_dir,
        "instruct_output_dir": instruct_output_dir
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate logps for reward and instruct model")
    parser.add_argument('--reward_model_name', type=str, default="RLAIF-V-7B")
    parser.add_argument('--reward_model_path', type=str, default="/data/yaoshu/models/RLAIF-V-7B")
    parser.add_argument('--instruct_model_name', type=str, default="RLAIF-V-12B")
    parser.add_argument('--instruct_model_path', type=str, default="/data/yaoshu/models/RLAIF-V-12B")
    parser.add_argument('--dataset_path', type=str, default='/data/yaoshu/dataset/RLAIF-V-Dataset')
    parser.add_argument('--reward_model_output_dir', type=str, default='/data/RLAIF-V-CC/results/reward')
    parser.add_argument('--instruct_model_output_dir', type=str, default='/data/RLAIF-V-CC/results/instruct')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl', world_size=int(os.getenv('WORLD_SIZE', '1')),
                            rank=int(os.getenv('RANK', '0')), )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    files = main(args.reward_model_name, args.reward_model_path, args.instruct_model_name, args.instruct_model_path,
                 args.dataset_path, args.reward_model_output_dir,
                 args.instruct_model_output_dir)
    print(files)

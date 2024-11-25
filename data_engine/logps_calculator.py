import os
import tqdm
import itertools
import argparse
from muffin.eval.muffin_inference_logp import (get_batch_logps, write_logp_to_preference_parquet)
from util import *

import torch
import torch.distributed as dist


def get_multimodal_sample_logps(model, dataloader, is_llava15=False):
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        idx = 0
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


def inference_logp(
        model_name,
        model_path,
        dataset_path,
        output_dir):
    """
    Args:
        model_name:  e.g. llava-v1.5-7, OmniLMM-12B, RLAIF-V-12B
        model_path: path to your model
        dataset_path: path to dataset(should follow RLAIF-V-Dataset format)
        output_dir: path to outputfile(logps)

    Returns:

    """

    model, dataset, dataloader = load_model_and_dataloader(model_path, model_name, dataset_path)

    outputs = get_multimodal_sample_logps(
        # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list
        model,
        dataloader,
        is_llava15=("llava" in model_name.lower() or (
                "rlaif" in model_name.lower() and "7b" in model_path.lower())))  # judge if the model follow llava structure

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]

    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list \
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list,
                     rej_per_token_logp_list))

    df = write_logp_to_preference_parquet(dataset.data, output_dir, logps, overwrite_logps=True)

    torch.distributed.barrier()

    del model
    return df


def main(
        reward_model_name: str,
        reward_model_path: str,
        instruct_model_name: str,
        instruct_model_path: str,
        dataset_path: str,
        reward_output_dir: str,
        instruct_output_dir: str):
    _ = inference_logp(instruct_model_name, instruct_model_path, dataset_path, instruct_output_dir)
    _ = inference_logp(reward_model_name, reward_model_path, dataset_path, reward_output_dir)

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

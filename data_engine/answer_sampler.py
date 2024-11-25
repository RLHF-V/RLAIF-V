import os
import tqdm
import copy
from chat import RLAIFVChat
from datasets import load_dataset
import torch
import pandas as pd
from muffin.data.datasets import bytes_to_PIL_image
from util import *
from collections import defaultdict
import torch.distributed as dist


def sample_answer(model_path, dataset_path, output_path, image_column, sample=10):
    # here we need to keep different samples of the same question adjacent to each other in the final file
    # otherwise, the data_pair_builder will output data with no sense.
    # so in this function, there are some code used to keep the order
    # if you want to change them, you may also need to change code in data_pair_builder

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        model = RLAIFVChat(model_path)
        grouped_output_data = defaultdict(list)

        with torch.inference_mode():
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }

            dataset = load_dataset(dataset_path, cache_dir='./cache')['train'].cast_column(
                image_column,
                hf_datasets.Image(decode=False)
            )

            total_size = len(dataset)
            base_size = total_size // world_size
            remainder = total_size % world_size

            start_idx = local_rank * base_size + min(local_rank, remainder)
            end_idx = start_idx + base_size + (1 if local_rank < remainder else 0)

            device_dataset = dataset.select(range(start_idx, end_idx))
            processed_indices = set()

            iterator = tqdm.tqdm(
                device_dataset,
                desc=f"GPU {local_rank}",
                position=local_rank
            )

            for idx, data in enumerate(iterator):
                try:
                    data_id = start_idx + idx
                    current_samples = []
                    for i in range(sample):
                        try:
                            data_cp = copy.deepcopy(data)
                            # your dataset should keep image in ['image']['bytes'] or ['image_bytes']['bytes']
                            # or you can change the following code to read the data in your format
                            if 'image' in data_cp:
                                data_cp['image'] = bytes_to_PIL_image(data_cp['image']['bytes'])
                                output = model.chat(data_cp, param=generation_config)
                                data_cp['chosen'] = output
                                data_cp['rejected'] = output
                                data_cp['image'] = data['image']
                                data_cp['global_index'] = data_id  # 添加全局索引
                                data_cp['sample_index'] = i  # 添加样本索引
                            elif 'image_bytes' in data_cp:
                                data_cp['image'] = bytes_to_PIL_image(data_cp['image_bytes']['bytes'])
                                output = model.chat(data_cp, param=generation_config)
                                data_cp['chosen'] = output
                                data_cp['rejected'] = output
                                data_cp.pop('image')
                                data_cp['image'] = data['image_bytes']
                                data_cp['global_index'] = data_id
                                data_cp['sample_index'] = i
                            else:
                                raise ValueError("image attribute not found")
                            current_samples.append(data_cp)
                        except Exception as e:
                            print(f"Error processing sample {i} for data_id {data_id}: {str(e)}")
                            continue

                    if current_samples:  # 只有在成功生成样本时才添加
                        grouped_output_data[data_id] = current_samples
                        processed_indices.add(data_id)
                except Exception as e:
                    print(f"Error processing data_id {data_id}: {str(e)}")
                    continue

            torch.distributed.barrier()

            if world_size > 1:
                all_data = [None] * world_size
                dist.all_gather_object(all_data, grouped_output_data)

                if local_rank == 0:
                    merged_data = defaultdict(list)
                    all_data_ids = set()
                    for rank_data in all_data:
                        all_data_ids.update(rank_data.keys())

                    for data_id in sorted(all_data_ids):
                        for rank_data in all_data:
                            if data_id in rank_data:
                                merged_data[data_id].extend(rank_data[data_id])
                    grouped_output_data = merged_data

            if local_rank == 0:
                step = 5000
                flat_output_data = []

                for data_id in sorted(grouped_output_data.keys()):
                    samples = sorted(grouped_output_data[data_id], key=lambda x: x['sample_index'])
                    flat_output_data.extend(samples)

                # 分批保存数据时保持顺序
                for idx, start in enumerate(range(0, len(flat_output_data), step)):
                    try:
                        temp_data = flat_output_data[start: min(start + step, len(flat_output_data))]
                        df = pd.DataFrame(temp_data)

                        df = df.sort_values(['global_index', 'sample_index'])
                        df = df.drop(columns=['global_index', 'sample_index'])

                        output_file = os.path.join(
                            output_path,
                            f'RLAIF-V-Dataset-sampled_{idx:03}-{len(temp_data)}.parquet'
                        )

                        temp_file = output_file + '.tmp'
                        df.to_parquet(temp_file)
                        os.rename(temp_file, output_file)

                    except Exception as e:
                        print(f"Error saving batch {idx}: {str(e)}")
                        continue

    except Exception as e:
        print(f"Critical error in sample_answer: {str(e)}")
        raise
    finally:
        if 'model' in locals():
            del model


def main():
    dist.init_process_group(backend='nccl')

    model_path = "your_model_path"
    dataset_path = "your_dataset_path"
    output_path = "your_output_path"
    sample = 10

    try:
        sample_answer(model_path, dataset_path, output_path, sample)
    finally:
        # 清理分布式环境
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

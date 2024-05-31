import os
import sys
from llava.model import *
import gc
import torch
import random
import logging
import copy
import pathlib
import getpass
import transformers
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from utils.utils import is_main_process, get_rank
from muffin.train.trainers import LLaVA15DPOTrainer
from muffin.data.datasets import SingleDataSourceDataset, MultiDataSourceDataset,RLAIFVDataset
from muffin.data.data_processors import register_data_path
from muffin.train.train_utils import encode_multimodal_preference_sample, preprocess_v1

from muffin.train.train_muffin import DataCollatorForDPODataset
from functools import partial
import muffin.conversation as conversation_lib

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="llava_v1") # llava 1.5
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'
    eval_data_source_names: Optional[str] = field(default=None)
    data_dir: str = './RLAIF-V-Dataset/'
    kto_win_data_source_names: str = '100'
    kto_win_data_source_weights: str = '100'
    kto_rej_data_source_names : str = '100'
    kto_rej_data_source_weights: str = '100'

    dpo_beta: float = 0.5
    dpo_token_weight: float = 3.0

    shuffle_data: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_steps: int = field(default=1_000)
    no_randaug: bool = False

    task: str = field(
        default='LM',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_use_average: bool = False
    dpo_token_weighted: bool = False

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    fully_tune: bool = False

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def create_multi_data_source_dataset(data_source_names, data_source_weights, shuffle=False):
    ds_list = []
    for name in data_source_names:
        ds = SingleDataSourceDataset(name, *register_data_path[name](), shuffle=shuffle)
        ds_list.append(ds)
    ds = MultiDataSourceDataset(ds_list, data_source_weights)
    return ds


class DPODataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_dir: str,
                 multimodal_cfg: dict,
                 reference_model = None):
        super(DPODataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = RLAIFVDataset(data_dir, reference_model, tokenizer,multimodal_cfg['image_token_len'], multimodal_cfg['image_processor'], multimodal_cfg['use_im_start_end'], is_llava15=True)
        self.multimodal_cfg = multimodal_cfg
        self.multimodal_cfg['keep_image_tag'] = True

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        preprocess_func = partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            source, self.tokenizer, self.multimodal_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict


def make_dpo_data_module(tokenizer, data_args,reference_model):
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_dir=data_args.data_dir,
                               multimodal_cfg=dict(
                                   is_multimodal=data_args.is_multimodal,
                                   image_token_len=data_args.image_token_len,
                                   image_folder=data_args.image_folder,
                                   image_aspect_ratio=data_args.image_aspect_ratio,
                                   use_im_start_end=getattr(
                                       data_args, 'mm_use_im_start_end', False),
                                   image_processor=getattr(
                                       data_args, 'image_processor', None),
                                   data_source_names=getattr(
                                       data_args, 'data_source_names'),
                                   data_source_weights=getattr(data_args, 'data_source_weights'),
                                   shuffle_data=data_args.shuffle_data
                                   ),
                               reference_model=reference_model)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)

    if data_args.eval_data_source_names is not None:
        eval_datasets = {}
        for name in data_args.eval_data_source_names:
            eval_dataset = DPODataset(tokenizer=tokenizer,
                                      data_dir=data_args.data_dir,
                                      multimodal_cfg=dict(
                                          is_multimodal=data_args.is_multimodal,
                                          image_token_len=data_args.image_token_len,
                                          image_folder=data_args.image_folder,
                                          image_aspect_ratio=data_args.image_aspect_ratio,
                                          use_im_start_end=getattr(
                                              data_args, 'mm_use_im_start_end', False),
                                          image_processor=getattr(
                                              data_args, 'image_processor', None),
                                          data_source_names=[name],
                                          data_source_weights=[1],
                                           shuffle_data=False
                                          ),
                                      reference_model=reference_model)
            eval_datasets[name] = eval_dataset
    else:
        eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)


def init_model(model_args, data_args, training_args, attn_implementation):
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        # attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None)
    )
    model.config.use_cache = False
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        truncation_side='right',
    )
    # for llava 1.5
    tokenizer.pad_token = tokenizer.unk_token
    assert model_args.version == 'llava_v1'
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        print("conv template:", conversation_lib.default_conversation)
    else:
        raise NotImplementedError

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.fully_tune:
        model.requires_grad_(True)

    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad]
    if is_main_process():
        print(f'No grad params are : {params_no_grad}', flush=True)

    if training_args.task == 'LM':
        raise NotImplementedError
    elif training_args.task == 'DPO':
        data_module = make_dpo_data_module(tokenizer, data_args=data_args, reference_model=copy.deepcopy(model).cuda())

    return model.cuda(), data_module, tokenizer


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.report_to == 'wandb':
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(['.cache', '_temp'])

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [
        int(x) for x in data_args.data_source_weights.split('#')]

    data_args.eval_data_source_names = data_args.eval_data_source_names.split(
        '#') if data_args.eval_data_source_names is not None else None

    data_args.kto_win_data_source_names = data_args.kto_win_data_source_names.split('#')
    data_args.kto_win_data_source_weights = list(map(int, data_args.kto_win_data_source_weights.split('#')))
    data_args.kto_rej_data_source_names = data_args.kto_rej_data_source_names.split('#')
    data_args.kto_rej_data_source_weights = list(map(int, data_args.kto_rej_data_source_weights.split('#')))

    model, data_module, tokenizer = init_model(
        model_args, data_args, training_args, attn_implementation=attn_implementation)

    if training_args.task == 'LM':
        raise NotImplementedError
    elif training_args.task == 'DPO':
        # TODO
        trainer = LLaVA15DPOTrainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   **data_module)

    # print(f'Training args: {training_args}')
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'Resume from checkpoint.')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'Train from start.')
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

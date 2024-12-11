#    Copyright 2023 Haotian Liu
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

from transformers import BitsAndBytesConfig
import torch


class ModelBuilder:
    @classmethod
    def judge_able_to_build(cls, model_name: str) -> bool:
        """
        Judge if the model can be built by this builder.
        Args:
            model_name: The name of the model.

        Returns:
            bool: True if the model can be built by this builder.
        """
        raise NotImplementedError

    @classmethod
    def build(cls, model_path, model_base, model_name, **kwargs):
        """
        Build the model.
        Returns:
            tokenizer: The tokenizer of the model.
            model: The model. This one must be returned. Otherwise, an error will be raised.
            image_processor: The image processor.
        """
        raise NotImplementedError


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # import here to avoid circular import
    from .llava_builder import LLaVABuilder
    from .omnillm_builder import OmniLLMBuilder
    from .minicpm_v_2_6 import MiniCPMV26Builder
    from .language_model_builder import LanguageModelBuilder

    # Note: please put LanguageModelBuilder at the end of the list if you want you add your own builder
    model_builder_list = [LLaVABuilder, OmniLLMBuilder, MiniCPMV26Builder, LanguageModelBuilder]

    tokenizer, model, image_processor = None, None, None
    for builder in model_builder_list:
        if builder.judge_able_to_build(model_name):
            tokenizer, model, image_processor = builder.build(model_path, model_base, model_name, **kwargs)
            break

    if model is None:
        raise ValueError(f"Cannot find a suitable builder for model {model_name}\n Please check whether the model name\
         is correct. If the model you use is not supported by default, please implement a new builder and add to the \
         model_builder_list in the file RLAIF-V/builder/builder.py")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

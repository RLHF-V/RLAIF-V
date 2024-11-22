from builder.builder import ModelBuilder

from transformers import AutoTokenizer
import torch
from omnilmm.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform


class OmniLLMBuilder(ModelBuilder):
    @classmethod
    def judge_able_to_build(cls, model_name: str) -> bool:
        lower_name = model_name.lower()
        return 'omnillm' in lower_name or ('rlaif' in lower_name and '12b' in lower_name)

    @classmethod
    def build(cls, model_path, _, model_name, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)

        if False:
            # model on multiple devices for small size gpu memory (Nvidia 3090 24G x2)
            with init_empty_weights():
                model = OmniLMMForCausalLM.from_pretrained(model_name, tune_clip=True, torch_dtype=torch.bfloat16)
            model = load_checkpoint_and_dispatch(model, model_name, dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 no_split_module_classes=['Eva', 'MistralDecoderLayer', 'ModuleList',
                                                                          'Resampler']
                                                 )
        else:
            model = OmniLMMForCausalLM.from_pretrained(
                model_path, tune_clip=True, torch_dtype=torch.bfloat16
            ).to(device='cuda', dtype=torch.bfloat16)

        img_processor = build_transform(
            is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN,
                              DEFAULT_IM_END_TOKEN], special_tokens=True)
        vision_config = model.model.vision_config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        return tokenizer, model, img_processor

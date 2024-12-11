import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

from builder.builder import ModelBuilder
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map


class MiniCPMV26Builder(ModelBuilder):
    @classmethod
    def judge_able_to_build(cls, model_name: str) -> bool:
        lower_name = model_name.lower()
        return ("minicpm-v" in lower_name or "minicpm_v" in lower_name) and (
                    "2_6" in lower_name or "2.6" in lower_name or "26" in lower_name)

    @classmethod
    def build(cls, model_path, model_base, model_name, **kwargs):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = model.to(torch.device("cuda"))

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.add_tokens(['<|im_end|>', '<|endoftext|>'], special_tokens=True)

        return tokenizer, model, image_processor

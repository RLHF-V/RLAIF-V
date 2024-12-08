from builder.builder import ModelBuilder
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LanguageModelBuilder(ModelBuilder):
    """
    **Note**: Please put this class at the end of the model builder list.
    This model builder is a fallback builder for all language models.
    It returns no image processor.
    """

    @classmethod
    def judge_able_to_build(cls, model_name: str) -> bool:
        return True

    @classmethod
    def build(cls, model_path, model_base, model_name, **kwargs):
        warnings.warn(
            "Warning: LanguageModel is the fall back model. Please make sure you are loading the correct model.")
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                                             **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        return tokenizer, model, None

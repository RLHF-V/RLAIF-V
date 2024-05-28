import io
import re
import glob
import math
import json
import base64
import random
import copy

from PIL import Image
from typing import List


class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(keys, value):
            if not callable(value):
                raise Exception(
                    f"Register object must be callable! But receice:{value} is not callable!")

            if not isinstance(keys, list):
                keys = [keys]

            for key in keys:
                if key in self._dict:
                    print(
                        f"error: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
                    exit()

                self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


register_data_processor = Register()
register_data_path = Register()

def b64_to_PIL_image(img_b64_buffer):
    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image


def wrap_ocr_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, ocr_instruction_templates)


def wrap_caption_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, caption_instruction_templates)


def gather_data_files_by_glob(root: str, pattern='*.tsv'):
    filenames = []

    for fullpath in glob.glob(f'{root}/{pattern}'):
        filename = fullpath.split('/')[-1]
        filenames.append(filename)
    return root, filenames


### llava1.5 7b auto=alignment, 1iter
@register_data_processor(['sr_llava15_llava15base_rmllava16_data_base_eq4000imgs'])
def dpo_data_processor(*args, **kwargs):
    return dpo_preference_processor(*args, **kwargs)

@register_data_path('sr_llava15_llava15base_rmllava16_data_base_eq4000imgs')
def dpo_data_path():
    data_dir = "data/train/"
    return gather_data_files_by_glob(data_dir, pattern='sr_omni_merge_diff1_samp2_llava15_checkby_llava16_2000detail#2500qaimgs_filtershort_0_logp-5998_sample_img88.9%-5342.tsv')

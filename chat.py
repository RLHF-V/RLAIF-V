import json

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates
from builder.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from PIL import Image
import base64
import io
import os
from omnilmm.train.train_utils import omni_preprocess


def init_omni_lmm(model_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load RLAIF-V-12B model and tokenizer from {model_name}')
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer


def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text[0]['content'] = im_st_token + im_patch_token * \
                                      image_token_len + im_ed_token + '\n' + question_text[0]['content']
    return question_text


def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    if isinstance(question, str):
        question = [{"role": "user", "content": question}]

    question = expand_question_into_multimodal(
        question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

    conversation = question
    data_dict = omni_preprocess(sources=[conversation],
                                tokenizer=tokenizer,
                                generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0],
                     labels=data_dict["labels"][0])
    return data_dict


class RLAIFV12B:
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()

    def decode(self, image, input_ids):
        with torch.inference_mode():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=1024,
                num_beams=3,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=30,
                top_p=0.9,
            )

            response = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True)
            response = response.strip()
            return response

    def chat(self, input):
        im_64 = img2base64(input['image'])
        msgs = json.dumps([{"role": "user", "content": input['question']}])

        try:
            image = Image.open(io.BytesIO(base64.b64decode(im_64))).convert('RGB')
        except Exception as e:
            return "Image decode error"

        msgs = json.loads(msgs)
        input_ids = wrap_question_for_omni_lmm(
            msgs, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.as_tensor(input_ids)
        image = self.image_transform(image)

        out = self.decode(image, input_ids)

        return out


def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string


class RLAIFV7B:
    def __init__(self, model_path) -> None:
        disable_torch_init()
        model_name = 'llava-v1.5-7b'
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, model_base=None, model_name=model_name, device_map={"": 'cuda'})
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

    def chat(self, input):
        msgs = input['question']
        if self.model.config.mm_use_im_start_end:
            msgs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + msgs
        else:
            msgs = DEFAULT_IMAGE_TOKEN + '\n' + msgs

        image = Image.open(input['image']).convert('RGB')
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], msgs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                temperature=0,
                num_beams=3,
                max_new_tokens=1024,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


class RLAIFVChat:
    def __init__(self, model_path) -> None:
        if '12B' in model_path:
            self.model = RLAIFV12B(model_path)
        else:
            self.model = RLAIFV7B(model_path)

    def chat(self, input):
        return self.model.chat(input)


if __name__ == '__main__':
    chat_model = RLAIFVChat('RLAIF-V/RLAIF-V-7B')  # or 'HaoyeZhang/RLAIF-V-12B'
    image_path = "./examples/test.jpeg"
    msgs = "Why did the car in the picture stop?"
    inputs = {"image": image_path, "question": msgs}
    answer = chat_model.chat(inputs)
    print(answer)

import os
import torch
import json
import copy
from PIL import Image
import base64
import argparse
import io
import tqdm
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel
import torch.utils.data as torch_data

from minicpmv_diverse_gen import MiniCPMVQADataset


class MiniCPM_Llama3_V_RM:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()
        self.config = self.model.config

    def raw_generate(
        self,
        input_id_list=None,
        img_list=None,
        tgt_sizes=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        **kwargs
    ):
        assert input_id_list is not None
        bs = len(input_id_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self.model._process_list(tokenizer, input_id_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(img.to(self.model.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.model.get_vllm_embedding(model_inputs)

            result = self._raw_decode(model_inputs["inputs_embeds"], tokenizer, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result

    def _raw_decode(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        output = self.model.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            **kwargs
        )
        return output

    def chat_with_scores(
        self,
        sample,
        vision_hidden_states=None,
        max_new_tokens=1,
        max_inp_length=2048,
        **kwargs
    ):

        msgs = sample['question']
        image = sample['image']

        copy_msgs = copy.deepcopy(msgs)
        assert len(copy_msgs) > 0, 'msgs is empty'

        if image is not None and isinstance(copy_msgs[0]['content'], str):
            copy_msgs[0]['content'] = [image, copy_msgs[0]['content']]

        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]

            images = []
            tgt_sizes = []
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    image = c
                    if self.config.slice_mode:
                        slice_images, image_placeholder = self.model.get_slice_image_placeholder(
                            image, self.tokenizer
                        )
                        cur_msgs.append(image_placeholder)
                        for slice_image in slice_images:
                            slice_image = self.model.transform(slice_image)
                            H, W = slice_image.shape[1:]
                            images.append(self.model.reshape_by_patch(slice_image))
                            tgt_sizes.append(torch.Tensor([H // self.config.patch_size, W // self.config.patch_size]).type(torch.int32))
                    else:
                        images.append(self.model.transform(image))
                        cur_msgs.append(
                            self.tokenizer.im_start
                            + self.tokenizer.unk_token * self.config.query_num
                            + self.tokenizer.im_end
                        )
                elif isinstance(c, str):
                    cur_msgs.append(c)

            if tgt_sizes:
                tgt_sizes = torch.vstack(tgt_sizes)

            msg['content'] = '\n'.join(cur_msgs)

        input_ids = self.tokenizer.apply_chat_template(copy_msgs, tokenize=True, add_generation_prompt=False)

        generation_config = {
            "num_beams": 1,
            "repetition_penalty": 1.2,
            "output_scores": True,
            "return_dict_in_generate": True
        }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        with torch.inference_mode():
            res = self.raw_generate(
                input_id_list=[input_ids],
                max_inp_length=max_inp_length,
                img_list=[images],
                tgt_sizes=[tgt_sizes],
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=False,
                **generation_config
            )

        yes_id = self.tokenizer.encode(f'{self.tokenizer.bos_token}yes')[-1]
        Yes_id = self.tokenizer.encode(f'{self.tokenizer.bos_token}Yes')[-1]
        no_id = self.tokenizer.encode(f'{self.tokenizer.bos_token}no')[-1]
        No_id = self.tokenizer.encode(f'{self.tokenizer.bos_token}No')[-1]

        print("output_ids:", res.sequences[0])
        print("response:", self.tokenizer.decode(res.sequences[0]))

        response = self.model._decode_text(res.sequences, self.tokenizer)
        # response = self.tokenizer.decode(
        #     res.sequences[0], skip_special_tokens=True)
        response = response[0].strip()

        output_scores = res.scores[0][0]
        scores = torch.softmax(output_scores, dim=0)
        print(scores.shape)
        max_value, max_index = torch.max(scores, dim=0)
        print(f'scores: {max_index}')

        item_scores = {
            'yes': scores[yes_id].cpu().item(),
            'Yes': scores[Yes_id].cpu().item(),
            'no': scores[no_id].cpu().item(),
            'No': scores[No_id].cpu().item()
        }

        return response, item_scores

def eval_autocheck(args):
    model = MiniCPM_Llama3_V_RM(args.model_name)

    answer_dir = '/'.join(args.answers_file.split("/")[:-1])
    print(answer_dir)
    assert os.path.exists(answer_dir), f'Expecting {answer_dir} to be existing'
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    qa_dataset = MiniCPMVQADataset(args.question_file, start=args.start, end=args.end, repeat=args.repeat, chunk_num=args.chunk_num, chunk_idx=args.chunk_idx)

    ans_file = open(answers_file, "w")

    with torch.inference_mode():
        for batch in tqdm.tqdm(qa_dataset, f'Generating answers'):
            response, score = model.chat_with_scores(batch) # temperature=args.temperature

            if 'ds_question_id' in batch['metainfo']:
                ans_file.write(json.dumps({
                    'question_id': batch['question_id'],
                    'ds_question_id': batch['metainfo']['ds_question_id'],
                    'raw_question': batch['raw_question'],
                    'answer': response,
                    'scores': score,
                    'metainfos': batch['metainfo'],
                    'model_path': args.model_name
                }) + "\n")
            else:
                ans_file.write(json.dumps({
                    'question_id': batch['question_id'],
                    'raw_question': batch['raw_question'],
                    'answer': response,
                    'metainfos': batch['metainfo'],
                    'model_path': args.model_name
                }) + "\n")

            ans_file.flush()
    ans_file.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--question-file", type=str,
                        default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--chunk-num", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    eval_autocheck(args)
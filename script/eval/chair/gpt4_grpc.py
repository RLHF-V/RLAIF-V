import os
import sys
import time
import json
import requests
import traceback

import openai


# openai.api_base = "https://api.zhiyungpt.com/v1"
# openai.api_key = 'sk-J3oUW0OZtUSW0NNm5a7268Ae9f2b437b933cA702A0714436'

# openai.api_base = 'https://yeysai.com/v1'
# openai.api_key = 'sk-C5u2D1eTh3iC33VS9f1c2bFdB5C04260B34966221a556b84'

openai.api_base='https://cn2us02.opapi.win/v1'
openai.api_key = 'sk-xIXHIDN8508485013120T3BLbKFJa099F1D19e3641a2bC33'

class Chat:
    def __init__(self, model="", timeout_sec=20, use_mianbi=True, use_hk=False):
        self.model = model
        self.timeout = timeout_sec
        self.use_mianbi = use_mianbi
        self.use_hk = use_hk

    def chat_completion(self, messages, temperature=0.2, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0):

        if self.use_mianbi:
            if "gpt-4" in self.model:
                response = requests.post("http://120.92.10.46:8080/chat", json={
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                }, timeout=self.timeout).json()

            else:
                response = requests.post("http://47.254.22.102:8989/chat", json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                }, timeout=self.timeout).json()
        elif self.use_hk:
            response = requests.post('https://api.openai-hk.com/v1/chat/completions',
                                     headers={
                                         "Content-Type": "application/json",
                                         "Authorization": "Bearer hk-sxx1clga8acagad5xfyh20lxzwmkm1gjs9myym13icwbcv5e"
                                     },
                                     data=json.dumps({
                                         "max_tokens": max_tokens,
                                         "model": self.model,
                                         "temperature": temperature,
                                         "top_p": top_p,
                                         "presence_penalty": 0,
                                         "messages": messages
                                     }).encode('utf-8')
                                     ).json()

        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=0,
                frequency_penalty=0,
                timeout=20,
            )

        return response


def get_eval(chat, content,
             chat_gpt_system='You are a helpful and precise assistant for checking the quality of the answer.',
             max_tokens=256,
             fail_limit=100,
             temperature=0.2,
             top_p=1.0,
             omit_version=False):
    fail_cnt = 0
    while True:
        try:
            resp = chat.chat_completion(
                messages=[
                    {"role": "system", "content": chat_gpt_system},
                    {"role": "user", "content": content}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            try:
                if resp['model'] != chat.model and not omit_version:
                    real_model = resp['model']
                    print(
                        f'Except {chat.model}, but got message from {real_model}', flush=True)
                    continue
                rtn = resp["choices"][0]["message"]["content"]
                # time.sleep(5)
                return rtn
            except:
                print(f'Response: {resp}')
        except Exception as e:
            print(e)
        fail_cnt += 1
        if fail_cnt == fail_limit:
            return f'-1\n<no_response>'
        time.sleep(10 + fail_cnt)


import os
import sys
import time
import glob
import pathlib
import json
import base64
import random
import requests
# import tiktoken
import jsonlines
import traceback

from tqdm import tqdm
from time import sleep
# from openai import OpenAI
from multiprocessing import Pool

import pandas as pd
import concurrent.futures

app_code = 'gpt_table_construction'
user_code = '4WImjNj5EngUs7w6RUksodBnJoxmex8WnXFLT2jGsx8'
# app_code = 'vlm_ocr_sft_cn'
# user_code = 'gH9Jc_6KeeRsaYFuZXOOLqano0j8wWudwAYdSIlIePA'

class ChatClient:
    def __init__(self, app_code=app_code, user_token=user_code, app_token=None):
        self.app_code = app_code
        self.user_token = user_token

        if app_token is not None:
            self.app_token = app_token
        else:
            self.app_token = self.get_app_token(app_code, user_token)

    def get_app_token(self, app_code, user_token):
        headers = {'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'}
        res = requests.get(
            f'https://llm-center.ali.modelbest.cn/llm/client/token/access_token?appCode={app_code}&userToken={user_token}&expTime=3600', headers=headers)
        assert res.status_code == 200
        js = json.loads(res.content)
        assert js['code'] == 0
        return js['data']

    def create_conversation(self, title='ocr_sft', user_id='tc'):
        url = 'https://llm-center.ali.modelbest.cn/llm/client/conv/createConv'
        headers = {
            'app-code': self.app_code,
            'app-token': self.app_token,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        data = {'title': title, 'userId': user_id, 'type': 'conv'}
        res = requests.request("POST", url, json=data, headers=headers)
        assert res.status_code == 200, f"status code: {res.status_code}"
        js = json.loads(res.content)
        assert js['code'] == 0
        return js['data']

    def chat_sync(self, system_prompt='You are a helpful assistant.', user_prompt='', base64_image='', conv_id=None, model_id=36, max_tokens=4096):
        # print("In system prompt:", system_prompt)
        # print("In user prompt:", user_prompt)
        print("model id:", model_id)

        url = 'https://llm-center.ali.modelbest.cn/llm/client/conv/accessLargeModel/sync'
        headers = {
            'app-code': self.app_code,
            'app-token': self.app_token,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        data = {
            'userSafe': 0,  # disable user safe
            'aiSafe': 0,
            'modelId': model_id,  # 15:GPT-4; 36: gpt4 1106 preview; 39; gpt4 vision preview; 32: gpt-3.5-turbo-1106
            'sysPrompt': system_prompt,
            'generateType': "NORMAL",
            'chatMessage': [
                {
                    "msgId": "",
                    "role": "USER",  # USER / AI
                    "contents": [
                        {
                            "type": "TEXT",
                            "pairs": user_prompt
                        },
                        {
                            "type": "IMAGE",
                            "pairs": f"data:image/jpg;base64,{base64_image}",
                        }
                    ],
                    "parentMsgId": "string",
                }
            ],
            "modelParamConfig": {
                "maxTokens": max_tokens,
                "temperature": 0.1,
            }
        }

        # drop empty image content, otherwise causing js['code'] == 0
        if not base64_image:
            data['chatMessage'][0]['contents'].pop(1)

        res = requests.request("POST", url, json=data, headers=headers)
        assert res.status_code == 200, f"status code: {res.status_code}"
        js = json.loads(res.content)
        assert js['code'] == 0, f'{str(js)}\n【{user_prompt}】'
        return js['data']['messages'][0]['content'], js

    def chat_sync_retry(self, system_prompt='You are a helpful assistant.', user_prompt='', base64_image='', conv_id=None, max_retry=3, model_id=36, max_tokens=4096):
        for i in range(max_retry):
            try:
                return self.chat_sync(system_prompt, user_prompt, base64_image, conv_id, model_id=model_id, max_tokens=max_tokens)
            except Exception as err:
                traceback.print_exc()
                print(err)
                time.sleep(3)
                self.app_token = self.get_app_token(
                    self.app_code, self.user_token)
        return None


if __name__ == '__main__':
    # get user_token from environment variable
    # user_token = os.environ.get('USER_TOKEN')
    # user_token = sys.argv[1]

    chat = ChatClient()
    res = chat.chat_sync_retry(user_prompt='你是数学老师', model_id=32, max_tokens=10)
    print(res)

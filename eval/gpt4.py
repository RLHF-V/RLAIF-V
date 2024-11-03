# run following command to install openai
# ```
# pip install openai==0.28
# ```

import time
import requests
import openai


openai.api_base =  'xxx'
openai.api_key =   'xxx'

class Chat:
    def __init__(self, model="", timeout_sec=20):
        self.model = model
        self.timeout = timeout_sec

    def chat_completion(self, messages, temperature=0.2, top_p=1, max_tokens=2048):
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
                return rtn, resp
            except:
                print(f'Response: {resp}')
        except Exception as e:
            print(e)
        fail_cnt += 1
        if fail_cnt == fail_limit:
            return f'-1\n<no_response>'
        time.sleep(10 + fail_cnt)

if __name__ == '__main__':
    chat = Chat(model='gpt-4-1106-preview')
    ans = get_eval(
        chat,
        chat_gpt_system="",
        content="what is your model version?",
        temperature=1e-5
    )
    print(ans)
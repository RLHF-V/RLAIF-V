import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5

class Chat:
    def __init__(self, model="", timeout_sec=20, openai_apikey=''):
        self.model = model
        self.timeout = timeout_sec
        openai.api_key = openai_apikey

    def chat_completion(self, messages, temperature=0.2, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0):

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )

        return response


def get_eval(model, content: str,
             chat_gpt_system='You are a helpful and precise assistant for checking the quality of the answer.',
             max_tokens: int=256,
             fail_limit=100,
             openai_apikey=""):

    openai.api_key = openai_apikey

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': chat_gpt_system
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            if response['model'] != model:
                real_model = response['model']
                print(f'Except f{model}, but got message from f{real_model}', flush=True)
                continue

            print(response['model'])

            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']
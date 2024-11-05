import os
import argparse
import json
import time
import jsonlines
from gpt4_grpc import Chat

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''

def read_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f1:
        for item in jsonlines.Reader(f1):
            data.append(item)
    return data

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(json_path, data, indent=4):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def merge_template_answer(data, org_data, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for i in range(len(org_data)):
        if 'answer' in data[i].keys():
            answer = data[i]['answer'].replace("Assistant:", "").strip()
        elif 'text' in data[i].keys():
            answer = data[i]['text'].replace("Assistant:", "").strip()
        elif 'model_answer' in data[i].keys():
            answer = data[i]['model_answer'].replace("Assistant:", "").strip()

        org_data[i]["model_answer"] = answer

    save_json(save_path, org_data)

    return org_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json', help='response file containing images, questions, and model responses')
    parser.add_argument('--evaluation', type=str, default=None, help='GPT-4 evaluation results to be saved')
    parser.add_argument('--api-key', type=str, required=True)
    parser.add_argument('--response-template', type=str, default="./eval/data/mmhal-bench_answer_template.json")
    parser.add_argument('--gpt-model', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--is_jsonl', action='store_true')
    args = parser.parse_args()

    print(args)

    if args.is_jsonl:
        org_data = read_json(args.response_template)
        data = read_jsonl(args.response)
        os.makedirs(os.path.dirname(args.response), exist_ok=True)
        records = merge_template_answer(data, org_data, f'{args.response}.template.json')

    else:
        # load json file, the json file should be in the same format as the response-template file
        with open(args.response, 'r') as f:
            records = json.load(f)

    assert len(records) == 96

    chat = Chat(model=args.gpt_model, timeout_sec=100, openai_apikey=args.api_key)

    # ask GPT-4 to evaluate
    responses = []
    for i, record in enumerate(records):
        image_content = ', '.join(record['image_content'])

        input_text = template.format(image_content, record['question'], record['gt_answer'], record['model_answer'])
        print(input_text)

        response = None
        while response is None:
            try:
                response = chat.chat_completion(
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0.0,
                )
                try:
                    if response['model'] != chat.model:
                        real_model = response['model']
                        print(f'Except f{chat.model}, but got message from f{real_model}', flush=True)

                        response = None
                        continue

                    print(response['model'])
                    content = response["choices"][0]["message"]["content"]
                    time.sleep(5)
                except:
                    print(f'Response: {response}')

            except Exception as e:
                print(e)
                print('retrying...')
                time.sleep(10)
                continue

        print(i, response['choices'][0]['message']['content'], flush=True)
        responses.append(response)
        time.sleep(1)

    # save responses
    if args.evaluation is not None:
        os.makedirs(os.path.dirname(args.evaluation), exist_ok=True)

        with open(args.evaluation, 'w') as f:
            json.dump(responses, f, indent=2)

    # merge gpt4 evaluation to answer
    for i in range(len(records)):
        records[i]["gpt4_review"] = responses[i]["choices"][0]["message"]["content"]

    os.makedirs(os.path.dirname(args.evaluation), exist_ok=True)
    with open(f'{args.evaluation}.merge_gpt4_score.json', "w") as f:
        json.dump(records, f, indent=4)

    # analyze responses
    scores = []
    for i, response in enumerate(responses):
        response = response['choices'][0]['message']['content']
        scores_found = []
        for s in range(7):
            if f'rating: {s}' in response.lower():
                scores_found.append(s)
        if len(scores_found) == 1:
            scores.append(scores_found[0])
        else:
            print('Warning: multiple or zero scores found')
            print(i, response)
            scores.append(0)

    hallucination = []
    for s in scores:
        if s >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)

    scores_each = [[] for _ in range(8)]
    # assuming order of 96 questions is not changed
    for i in range(96):
        question_type = i % 8
        scores_each[question_type].append(scores[i])

    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.3f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join([str(round(sum(scores_each[i]) / len(scores_each[i]), 2)) for i in range(8)]), flush=True)
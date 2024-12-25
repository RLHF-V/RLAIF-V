from copy import deepcopy

critic_prompt = "Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [{}]\nThe first response: [{}]\nThe second response: [{}]\nASSISTANT:\n"


def eval_builder(answers: list[dict[str, any]], sample_k=10):
    answers = deepcopy(answers)
    data = list(answers)
    answer_pairs = [data[i:i + sample_k] for i in range(0, len(data), sample_k)]

    for pairs in answer_pairs:
        for index, pair in enumerate(pairs):
            pair_inner_idx = index
            pair["inner_idx"] = pair_inner_idx
            question = pair["question"]
            response_1 = pair["chosen"]
            for compare in pairs:
                if compare == pair:
                    continue
                response_2 = compare["chosen"]
                prompt = critic_prompt.format(question, response_1, response_2)
                pair["critic_prompt"] = prompt

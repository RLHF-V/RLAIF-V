def judge_is_llava(model_name: str) -> bool:
    lower_name = model_name.lower()
    return 'llava' in lower_name or ('rlaif' in lower_name and '7b' in lower_name)


def judge_is_omnilmm(model_name: str) -> bool:
    lower_name = model_name.lower()
    return 'omnilmm' in lower_name or ('rlaif' in lower_name and '12b' in lower_name)

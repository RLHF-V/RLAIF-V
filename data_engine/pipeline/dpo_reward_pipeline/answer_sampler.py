import llava.llava15_sample_data
import omnilmm.omnilmm_sample_data
import minicpm_v_26.minicpmv_sample_data

from data_engine.util import *


def sample_answer(model_name, model_path, dataset_path, output_path, sample_k=10):
    if judge_is_llava(model_name=model_name):
        llava.llava15_sample_data.main(model_name, model_path, None, dataset_path, output_path, sample=sample_k,
                                       batch_size=sample_k)
        return
    if judge_is_omnilmm(model_name=model_name):
        omnilmm.omnilmm_sample_data.main(model_path, dataset_path, output_path, sample=sample_k, batch_size=sample_k)
        return
    if judge_is_minicpmv26(model_name=model_name):

        minicpm_v_26.minicpmv_sample_data.main(model_name, model_path, dataset_path, output_path, sample=sample_k,
                                               batch_size=sample_k)
        return

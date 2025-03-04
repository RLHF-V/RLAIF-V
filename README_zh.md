
<div align="center" style="font-size: 15pt">

<img src="examples/logo.png" width="30%" alt="RLAIF-V" />

**开源反馈实现超越 GPT-4V 的可信度**

<a href='https://arxiv.org/abs/2405.17220'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a>
<a href='https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset'><img src='https://img.shields.io/badge/Dataset-HF-Green'></a> <a href='https://huggingface.co/openbmb/RLAIF-V-7B'><img src='https://img.shields.io/badge/Model-7B-orange'></a> <a href='https://huggingface.co/openbmb/RLAIF-V-12B'><img src='https://img.shields.io/badge/Model-12B-orange'></a>

<h4 align="center">
    <p>
        <b>中文</b> | <a href="README.md">English</a>
    </p>
</h4>

</div>


## 🎊 更新日志 <!-- omit in toc -->

- [2025.03.01] 🎉 RLAIF-V 被 CVPR 2025 接收了，请大家关注我们最新版的[论文](https://arxiv.org/abs/2405.17220)!
- [2024.11.26] 🚀 我们现在支持使用 [LoRA](https://github.com/RLHF-V/RLAIF-V/blob/main/README_zh.md#%E8%AE%AD%E7%BB%83) 训练了！
- [2024.05.28] 📃 RLAIF-V 论文可以在 [arXiv](https://arxiv.org/abs/2405.17220) 访问了，欢迎进一步了解!
- [2024.05.20] 🔥 我们的 [RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) 数据集被用于第一个具有 GPT-4V 性能的端侧多模态大模型 [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) 的训练中！
- [2024.05.20] 我们开源了 RLAIF-V 的代码，权重（[7B](https://huggingface.co/openbmb/RLAIF-V-7B), [12B](https://huggingface.co/openbmb/RLAIF-V-12B)）和 [数据](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) !

## 📜 简介 <!-- omit in toc -->

我们提出了 RLAIF-V 框架，在完全开源的范式中对齐多模态大模型，并实现了超越 GPT-4V 的可信度。 RLAIF-V 从构造高质量反馈数据和应用在线反馈学习算法这两个关键角度最大限度地利用了开源反馈，其的显著特点包括：

* 💪 **通过开源反馈实现超越 GPT-4V 的可信度。** 通过从开源反馈中学习，RLAIF-V 12B 在生成任务和判别任务中都实现了超越 GPT-4V 的可信度。

<table align="center">
    <p align="center">
      <img src="examples/introduction1.png" width="80%" alt="introduction1" />
    </p>
</table>



* 🤝 **高质量的通用反馈数据。** RLAIF-V 使用的反馈数据可以**有效减少不同多模态大模型的幻觉**。

<table align="center">
    <p align="center">
      <img src="examples/introduction2.png" width="80%" alt="introduction2" />
    </p>
</table>

* 🚀 **对齐后模型指导的推理时增强。** 以 RLAIF-V 模型作为奖励模型可以在扩大推理预算时持续提高不同模型的可信度表现。

<table align="center">
    <p align="center">
      <img src="examples/introduction3.png" width="50%" alt="introduction3" />
    </p>
</table>


## 📌目录 <!-- omit in toc -->

- [数据集](#数据集)
- [安装](#安装)
- [模型权重](#模型权重)
- [推理](#推理)
- [数据生成](#数据生成)
- [训练](#训练)
- [评估](#评估)
  - [Object HalBench](#object-halbench)
  - [MMHal Bench](#mmhal-bench)
  - [RefoMB](#refomb)
- [引用](#引用)

## 数据集

我们提供了[RLAIF-V 数据集](https://huggingface.co/datasets/HaoyeZhang/RLAIF-V-Dataset)，这是一个由 AI 生成的偏好数据集，涵盖各种任务和领域。这一开源多模态偏好数据集包含3万多个高质量对比对。

## 安装

1. 克隆该仓库并进入 RLAIF-V 目录
```bash
git clone https://github.com/RLHF-V/RLAIF-V.git
cd RLAIF-V
```

2. 安装软件包
```bash
conda create -n rlaifv python=3.10 -y
conda activate rlaifv
pip install -e .
```
3. 安装所需的 spaCy 模型
```bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3.tar.gz
pip install en_core_web_trf-3.7.3.tar.gz
```


## 模型权重


| 模型|介绍     | 下载  |
|-----------------|------------------|:-------------:|
| RLAIF-V 7B   | 幻觉率最低的 LLaVA 1.5 版本 | [🤗](https://huggingface.co/openBMB/RLAIF-V-7B) |
| RLAIF-V 12B | 基于 OmniLMM-12B，实现超越 GPT-4V 的可信度。 | [🤗](https://huggingface.co/openBMB/RLAIF-V-12B)    |


## 推理

我们提供了一个简单的示例来说明如何使用 RLAIF-V。

```python

from chat import RLAIFVChat, img2base64

chat_model = RLAIFVChat('openBMB/RLAIF-V-7B')  # or 'openBMB/RLAIF-V-12B'
image_path="./examples/test.jpeg"
msgs = "Describe in detail the people in the picture."
inputs = {"image": image_path, "question": msgs}
answer = chat_model.chat(inputs)
print(answer)

```


您也可以通过执行以下脚本来运行此示例：

```bash
python chat.py
```

<details>
  <summary>
    <b>示例的输入和预期输出</b>
  </summary>



<div align="center">
<img src="examples/test.jpeg" width="500px">
</div>

**问题：**

Why did the car in the picture stop?

**预期输出：**

In the picture, a car stopped on the road due to the presence of a sheep on the roadway. The car likely stopped to allow the sheep to safely move out of the way or avoid any potential accidents with the animal. This situation highlights the importance of being cautious and attentive while driving, especially in areas where animals may roam near roads.

</details>


## 数据生成
1. 环境配置

我们提供 OmniLMM 12B 模型和 MiniCPM-Llama3-V 2.5 模型生成反馈的流程。如果您希望使用 MiniCPM-Llama3-V 2.5 提供反馈，请根据[MiniCPM-V GitHub仓库](https://github.com/OpenBMB/MiniCPM-o)中的说明配置其推理环境。

请下载我们微调的 Llama3 8B 模型：[分割模型](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_split_model.tar.gz)和[问题转换模型](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_changeq_model.tar.gz)，并将它们分别存储在`./models/llama3_split`文件夹和`./models/llama3_changeq`文件夹中。

2. OmniLMM 12B 模型反馈

以下脚本演示了如何使用 LLaVA-v1.5-7b 模型生成候选答案，并使用 OmniLMM 12B 模型提供反馈。

```bash
mkdir ./results
bash ./script/data_gen/run_data_pipeline_llava15_omni.sh
```

3. MiniCPM-Llama3-V 2.5 模型反馈

以下脚本演示了如何使用 LLaVA-v1.5-7b 模型生成候选答案，并使用 MiniCPM-Llama3-V 2.5 模型提供反馈。首先，将`./script/data_gen/run_data_pipeline_llava15_minicpmv.sh`中的`minicpmv_python`替换为您创建的 MiniCPM-V 环境的 Python 路径。

```bash
mkdir ./results
bash ./script/data_gen/run_data_pipeline_llava15_minicpmv.sh
```


## 训练

1. 数据准备

如果您可以访问huggingface数据集，您可以跳过这一步，我们将自动下载[RLAIF-V Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)。

如果您已经下载了数据集，您可以在[第38行](muffin/data/datassets.py#L38)，将`openbmb/RLAIF-V-Dataset`替换为您的数据集路径。

2. 开始训练
运行以下命令开始训练。

- **全参数训练**
```bash
bash ./script/train/llava15_train.sh
```

- **LoRA训练**
```bash
pip install peft
bash ./script/train/llava15_train_lora.sh
```
3. 迭代对齐

要重现论文中的迭代训练过程，您需要执行以下步骤4次：
- **S1. 数据生成**

  请遵循[数据生成](https://github.com/RLHF-V/RLAIF-V?tab=readme-ov-file#data-generation)中的说明为基础模型生成偏好对。将生成的 jsonl 文件转换为 huggingface parquet 格式。

- **S2. 修改训练配置**

  在数据集代码中，将[此处](muffin/data/datasets.py#L38)的`'openbmb/RLAIF-V-Dataset'`替换为您的数据路径。
  
  在[训练脚本](script/train/llava15_train.sh)中，将`--data_dir`替换为新目录，将`--model_name_or_path`替换为基础模型路径，将`--max_step`设置为 4 个周期的步数，将`--save_steps`设置为 1/4 个周期的步数。

- **S3. 进行 DPO 训练**

  运行训练脚本对基础模型进行训练。

- **S4. 选择新的基础模型进行下一次迭代**

  在Object HalBench和MMHal Bench上评估每个检查点模型，选择表现最佳的模型作为下一次迭代的基础模型。


## 评估


### Object HalBench

1. 准备 COCO2014 注释

Object HalBench 的评估依赖于 COCO2014 数据集中的图片说明和分割注释。请首先从 COCO 数据集的官方网站下载 COCO2014 数据集。

```bash
mkdir coco2014
cd coco2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip
```

2. 推理、评估和汇总

请将 `{YOUR_OPENAI_API_KEY}` 替换为有效的 OpenAI api-key。

**注: 评测时采用 `gpt-3.5-turbo-0613` 模型。**

```bash
# cd RLAIF-V

bash ./script/eval_rlaif_objhal.sh ./RLAIF-V_weight ./results/RLAIF-V ./coco2014/annotations {YOUR_OPENAI_API_KEY}
```


### MMHal Bench

1. 准备 MMHal 数据

请在[此处](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view?usp=sharing)下载 MMHal 评估数据，并将文件保存在`eval/data`中。

2. 运行以下脚本，生成、评估和汇总 MMHal Bench 的结果

**注: 评测时采用 `gpt-4-1106-preview` 模型。**

```bash
# cd RLAIF-V

bash ./script/eval_rlaifv_mmhal.sh ./RLAIF-V_weight ./results/RLAIF-V {YOUR_OPENAI_API_KEY}
```


### RefoMB

1. 准备环境与数据

使用 GPT-4 评估时，请首先运行`pip install openai==0.28`来安装 openai 包。接下来，将`eval/gpt4.py`中的`openai.base`和`openai.api_key`更改为您自己的设置。

验证集的评估数据位于 `eval/data/RefoMB_dev.jsonl`。您需要根据每行中的`image_url`键下载每张图片。

2. 评估总得分

将模型回复保存在`eval/data/RefoMB_dev.jsonl`数据中的`answer`键下，例如：

```
{
    "image_url": "https://thunlp.oss-cn-qingdao.aliyuncs.com/multimodal_openmme_test_20240319__20.jpg",
    "question": "What is the background of the image?",
    "type": "Coarse Perception",
    "split": "dev",
    "answer": "The background of the image features trees, suggesting that the scene takes place outdoors.",
    "gt_description": "......"
}
```
运行以下脚本对模型结果进行评估:

```
save_dir="YOUR SAVING DIR"
model_ans_path="YOUR MODEL ANSWER PATH"
model_name="YOUR MODEL NAME"

bash ./script/eval/run_refobm_overall.sh $save_dir $model_ans_path $model_name
```

3. 评估幻觉得分

评估总体得分后，将创建一个名为`A-GPT-4V_B-${model_name}.json`的评估结果文件。使用该评估结果文件计算幻觉分数，方法如下：

```
eval_result="EVAL RESULT FILE PATH, e.g. 'A-GPT-4V_B-${model_name}'"
# Do not include ".json" in your file path!

bash ./script/eval/run_refomb_hall.sh $eval_result
```

4. **注：** 为了取得更好的稳定性，我们建议使用 **3 次** 评估结果的 **平均值** 作为模型最终得分。


## 许可证 <!-- omit in toc -->


[![代码许可证](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![数据许可证](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**使用和许可声明**：数据、代码和模型仅供研究使用。它们也仅限于遵循 LLaMA、Vicuna 和 Chat GPT 许可协议的用途。数据集为 CC BY NC 4.0（仅允许非商业用途），使用该数据集训练的模型不得用于研究目的之外的用途。



## 致谢 <!-- omit in toc -->

- [RLHF-V](https://github.com/RLHF-V/RLHF-V): 本项目基于的代码库。
- [LLaVA](https://github.com/haotian-liu/LLaVA): RLAIF-V-7B的指令模型和标注模型。
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V): RLAIF-V-12B的指令模型和标注模型。
- [MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o): 应用 RLAIF-V 对齐训练的端到端全模态模型.

## 引用

如果您觉得我们的模型/数据/代码/论文有帮助，请给我们 ⭐ 和 引用 📝，感谢！

```bibtex
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}

@article{yu2024rlaifv,
  title={RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness},
  author={Yu, Tianyu and Zhang, Haoye and Yao, Yuan and Dang, Yunkai and Chen, Da and Lu, Xiaoman and Cui, Ganqu and He, Taiwen and Liu, Zhiyuan and Chua, Tat-Seng and Sun, Maosong},
  journal={arXiv preprint arXiv:2405.17220},
  year={2024},
}
```



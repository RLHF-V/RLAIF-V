
<div align="center" style="font-size: 15pt">

<img src="examples/logo.png" width="30%" alt="RLAIF-V" />

**通过开源反馈实现超越 GPT-4V 的可信度**

<a href='https://huggingface.co/datasets/HaoyeZhang/RLAIF-V-Dataset'><img src='https://img.shields.io/badge/Dataset-HF-Green'></a>
<!-- <a href=''><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> -->

<h4 align="center">
    <p>
        <b>中文</b> | <a href="README.md">English</a>
    </p>
</h4>

</div>


## 🎊更新日志 <!-- omit in toc -->

- [2024.05.20] 我们开源了RLAIF-V的代码，权重和数据！


## 📜简介 <!-- omit in toc -->

我们提出了一个新颖的框架RLAIF-V，它在完全开源的范式中对齐多模态大模型，实现了超越 GPT-4V 的可信度。 RLAIF-V 从两个关键角度最大限度地利用了开源反馈，包括高质量反馈数据和在线反馈学习算法。RLAIF-V 的显著特点包括：

* 💪 **通过开源反馈实现超越 GPT-4V 的可信度。** 通过从开源反馈中学习，RLAIF-V 12B 在生成任务和判别任务中都实现了超越 GPT-4V 的可信度。
  
<table align="center">
    <p align="center">
      <img src="examples/introduction1.png" width="80%" alt="introduction1" />
    </p>
</table>



* 🤝 **高质量的通用反馈数据。** RLAIF-V 使用的反馈数据可以**有效减少不同多模态大模型的幻觉**。

<table align="center">
    <p align="center">
      <img src="examples/introduction3.png" width="80%" alt="introduction3" />
    </p>
</table>


* ⚡️ **迭代对齐的高效反馈学习。** 与未采用迭代的方法相比，RLAIF-V 表现出**更高的学习效率和更好的性能**。

<table align="center">
    <p align="center">
      <img src="examples/introduction2.png" width="80%" alt="introduction2" />
    </p>
</table>


## 📌目录 <!-- omit in toc -->

- [数据集](#数据集)
- [安装](#安装)
- [模型权重](#模型权重)
- [推理](#推理)
- [训练](#训练)
- [评估](#评估)
  - [Object HalBench](#object-halbench)
  - [MMHalBench](#mmhalbench)

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


| 模型           | 描述                                                                | 下载链接                                                    |
|-----------------|----------------------------------------------------------------------------|--------------------------------------------------------------|
| **RLAIF-V-7B**  | 基于 LLaVA 1.5 7，提供高效推理。 | [🤗RLAIF-V-7B](https://huggingface.co/xiaomanlu/RLAIF-V-7B/) | 
| **RLAIF-V-12B** | 基于 OmniLMM-12B，实现超越 GPT-4V 的可信度。 | [🤗RLAIF-V-12B](https://huggingface.co/HaoyeZhang/RLAIF-V-12B)    |                                        


## 推理

我们提供了一个简单的示例来说明如何使用 RLAIF-V。

```python

from chat import RLAIFVChat, img2base64

chat_model = RLAIFVChat('RLAIF-V/RLAIF-V-7B')  # or 'HaoyeZhang/RLAIF-V-12B'
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




## 训练

1. 数据准备

下载 [RLAIF-V Dataset](https://huggingface.co/datasets/HaoyeZhang/RLAIF-V-Dataset) 到 `train/data` 目录下。如有需要，您可以在 [第101行](muffin/data/data_processors.py#L101) 修改数据集路径。

2. 开始训练
3. 运行以下命令开始训练。

```bash
bash ./script/train/llava15_train.sh
```

## 评估


### Object HalBench

1. 准备 COCO2014 注释

Object HalBench 的评估依赖于 COCO2014 数据集中的字幕和分割注释。请首先从 COCO 数据集的官方网站下载 COCO2014 数据集。

```bash
mkdir coco2014
cd coco2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip
```

2. 推理、评估和汇总

请将 `{YOUR_OPENAI_API_KEY}` 替换为有效的 OpenAI api-key。

```bash
# cd RLAIF-V

bash ./script/eval_rlaif_objhal.sh ./RLAIF-V_weight ./results/RLAIF-V ./coco2014/annotations {YOUR_OPENAI_API_KEY}
```


### MMHalBench

1. 准备 MMHal 数据

请在[此处](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view?usp=sharing)下载 MMHal 评估数据，并将文件保存在`eval/data`中。

2. 运行以下脚本，生成、评估和汇总 MMHal Bench 的结果
   
```bash
# cd RLAIF-V

bash ./script/eval_rlaifv_mmhal.sh ./RLAIF-V_weight ./results/RLAIF-V {YOUR_OPENAI_API_KEY}
```

 
## 许可证 <!-- omit in toc -->


[![代码许可证](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![数据许可证](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**使用和许可声明**：数据、代码和模型仅供研究使用。它们也仅限于遵循 LLaMA、Vicuna 和 Chat GPT 许可协议的用途。数据集为 CC BY NC 4.0（仅允许非商业用途），使用该数据集训练的模型不得用于研究目的之外的用途。



## 致谢 <!-- omit in toc -->

- [RLHF-V](https://github.com/RLHF-V/RLHF-V): 我们基于的代码库。
- [LLaVA](https://github.com/haotian-liu/LLaVA): RLAIF-V-7B的指令模型和标注模型。 
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V): RLAIF-V-12B的指令模型和标注模型。

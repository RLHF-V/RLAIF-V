# 数据引擎

## 概述

此模块用于构建用于直接训练的 DPO 数据集。只需输入您的Reward model、Instruct model和数据集，然后运行 `run_engine.sh` 脚本即可生成数据集。

- **Instruct model**：用于生成数据集中问题的初始答案。  
- **Reward model**：评估Instruct model生成的答案，提供奖励以对其排序并构建 DPO 数据集。

## 使用说明

参阅 `run_engine.sh` 脚本执行。数据构建支持以下两种pipeline：

1. **Divide and Conquer Pipeline**（`divide_and_conquer`）  
2. **DPO Reward Pipeline**（`dpo_reward`）  

两种pipeline的需求各有不同，具体如下。

---

### Divide and Conquer Pipeline

#### 处理方法

使用 RLAIF-V 分而治之策略收集 AI 反馈。

#### 所需模型

- **拆分模型**：[点击下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_split_model.tar.gz)  
- **问题转换模型**：[点击下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_changeq_model.tar.gz)  

#### 自定义实现

支持使用 `llava-1.5-7b` 作为Instruct model。Reward model需要以下三种模型：

1. **问题转换模型**（现支持rlaifv_llama3_changeq_model）
2. **问题拆分模型**（现支持rlaifv_llama3_split_model）
3. **自动检查模型**（现支持 `OmniLMM-12B` 或 `MiniCPM-Llama3-V-2_5`）  

对于问题转换模型和问题拆分模型，不建议更换。如需使用其他自动检查模型或Instruct model，请自定义实现以下内容：

1. **Sample Rollout**
    - 若要添加新的Instruct model则需要以下修改。
    - 在 `RLAIF-V/builder` 中添加模型构建器。  
    - 在 `RLAIF-V/builder/builder.py` 中添加对应调用代码。  
    - 参考 `RLAIF-V/muffin/llava15_gen_data.py` 实现采样逻辑。  

2. **Reward Collection**  
    - 若要添加新的自动检查模型则需要以下修改。
    - 如同生成 Rollout 一样，添加模型构建器和实用函数。  
    - 更新 `RLAIF-V/data_engine/pipeline/divide_and_conquer/divide_and_conquer_pipeline.py` 中的 `reward_calculate` 方法。

#### 数据集格式

数据集需为 `.jsonl` 格式，包含以下字段：

- `question`：与图像相关的问题。  
- `question_id`：可选字段。  
- `image`：Base64 编码的二进制数据（如果提供 `image_path` 则可省略）。  
- `image_path`：图像的路径（如果提供 `image` 则可省略）。如果 `image` 为空，将使用 `image_path`。

#### 脚本参数

- `--reward_model_path`：自动检查模型、问题转换模型和拆分模型的路径，使用逗号分隔（如：`/path/to/MiniCPM-Llama3-V-2_5,/path/to/changeq_model,/path/to/split_model`）。  
- `--reward_model_name`：自动检查模型名称（如：`MiniCPM-Llama3-V-2_5`）。  
- `--pipeline_name`：使用的pipeline名称，设置为 `divide_and_conquer`。  
- `--reward_model_python_path`：配置自动检查模型环境的 Python 路径（仅 MiniCPM-V 模型需要）。

---

### DPO Reward Pipeline

#### 处理方法

将 RLAIF-V 自反馈指导与 DPO 训练模型结合使用。

#### 自定义实现

支持的模型：

- Instruct model：`llava-1.5-7b`，`OmniLMM-12B`  
- Reward model：`RLAIF-V-7B`，`RLAIF-V-12B`  

如需使用其他模型，请实现以下自定义代码：

1. **Sample Rollout**  
    - 若要添加新的Instruct model则需要以下修改。
    - 在 `RLAIF-V/builder` 中添加模型构建器。  
    - 在 `RLAIF-V/builder/builder.py` 中添加调用代码。  
    - 参考 `RLAIF-V/llava/llava15_sample_data.py` 实现采样逻辑。  
    - 更新 `RLAIF-V/data_engine/pipeline/dpo_reward_pipeline/answer_sampler.py`。

2. **Reward Collection**  
    - 若要添加新的Reward model则需要以下修改。
    - 类似Sample Rollout添加模型构建器。  
    - 更新 `RLAIF-V/data_engine/pipeline/dpo_reward_pipeline/logps_calculator.py` 和 `RLAIF-V/muffin/eval/muffin_inference_logp.py`，以确保数据格式一致。

#### 数据集格式

推荐格式：`.parquet`  
字段：

- `idx`：每条记录的唯一标识符（可使用字符串）。  
- `question`：与图像相关的问题。  
- `image`：包含以下键的字典：  
    - `bytes`：二进制格式图像数据。  
    - `path`：可选字段，推荐提供。

#### 脚本参数

- `--pipeline_name`：设置为 `dpo_reward`。

---

### 通用使用说明

- 使用 `--work_dir` 指定工作目录，将中间和最终输出存储到该目录下。  
- 使用 `--debug` 参数，可将详细中间输出保存到 `debug` 目录中。  
- 如果出现错误，可使用 `--run_stage` 参数在解决问题后重新运行特定步骤。  
- 脚本完成后将显示最终数据集路径。

---

### 运行脚本

```bash
sh data_engine/run_data_engine.sh
```

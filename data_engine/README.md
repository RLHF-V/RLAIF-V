# Data Engine

## Welcome
Thank you for using Data Engine.  
This part of the code is used to build the DPO dataset, which you can use for direct training.  
You only need to input the reward model, instruction model, and your dataset, and we will generate the DPO dataset for you. All you need to do is run the `run_engine.sh` script.

## Usage
Please refer to the `run_engine.sh` script.

You will need to provide the path and name for both the reward model and the instruction model. Currently, we support the following models: llava-1.5-7b, RLAIF-V-7B, OmniLMM-12B, and RLAIF-V-12B. We are considering adding more models in the future. If the model you wish to use is not listed, you may need to implement the corresponding code yourself (for model loading, add code to `RLAIF-V/builder`; for data formatting, change `RLAIF-V/muffin/train/train_utils.py` and `RLAIF-V/data_engine/util.py`; for log probability calculation, change `RLAIF-V/data_engine/logps_calculator.py` and `RLAIF-V/muffin/eval/muffin_inference_logp.py`).

Additionally, **please double-check that the model name you provide is correct**, as we will not know which code to execute otherwise.

Next, your dataset should contain the following fields:
1. `idx`: A unique index for each data entry (this can be a string).
2. `question`: The question related to the image.
3. `image`: You can determine the column name by passing it via the `--image_column` parameter. The column should follow this structure:
   - `{'bytes': ..., 'path':...}`
   - `bytes` should be in binary format.
   - `path` is not strictly required, but to avoid errors, it's better to keep this field (you can set it as an empty string).
4. `image_path`: This field is optional; we will retain it in your final DPO dataset.
5. `ds_name`: This field is also optional; it will be retained in the final DPO dataset.
6. `origin_split`: This field is optional, but **if you pass it, please ensure it is in JSON format**. It will be retained in the final DPO dataset.

You can specify a `--work_dir` to store intermediate files and the final output under this directory (which will actually be a subdirectory within it).

If you encounter errors during generation, you can pass the stage next to the stage that has been completed using the `--continue_from_stage` parameter (0, 1, or 2). When the value is 0, it will start from scratch. (For example, if you've completed stages 0 and 1 but encounter an error during stage 2, you can fix the issue and set `--continue_from_stage 2` to continue from that point.) You can check the `data_engine.py` file for details on what each stage does.

Run:
```shell
sh data_engine/run_data_engine.sh
```

## Conclusion
If you run into any issues, feel free to contact us by submitting an Issue.

Thank you for choosing RLAIF-V. Best wishes for your project!

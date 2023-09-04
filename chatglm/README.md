# ChatGLM Efficient Tuning

## Datasets

Please refer to [data/README.md](data/README.md) for details.

## Fine-Tuning Methods

Our script now supports the following fine-tuning methods:

- [LoRA](https://arxiv.org/abs/2106.09685)
  - Fine-tuning the low-rank adapters of the model.
- [P-Tuning V2](https://github.com/THUDM/P-tuning-v2)
  - Fine-tuning the prefix encoder of te model.
- [Freeze](https://arxiv.org/abs/2012.14913)
  - Fine-tuning the MLPs in the last n blocks of the model.
- Full Tuning
  - Fine-tuning all the parameters of the model.

## Requirement

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers, Datasets, Accelerate, PEFT and TRL
- fire, protobuf, cpm-kernels and sentencepiece
- jieba, rouge-chinese and nltk (used at evaluation)
- gradio and matplotlib (used in train_web.py)
- uvicorn, fastapi and sse-starlette (used in api_demo.py)

And **powerful GPUs**!

## Getting Started

### Data Preparation (optional)

Please refer to `data/example_dataset` for checking the details about the format of dataset files. You can either use a single `.json` file or a [dataset loading script](https://huggingface.co/docs/datasets/dataset_script) with multiple files to create a custom dataset.

Note: please update `data/dataset_info.json` to use your custom dataset. About the format of this file, please refer to `data/README.md`.

### Dependence Installation (optional)

```bash
git lfs install
git clone https://github.com/THUDM/scholar-profiling.git
conda create -n sc_profiling_chatglm python=3.10
conda activate sc_profiling_chatglm
cd scholar-profiling/chatglm
pip install -r requirements.txt
```

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you will be required to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.1.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.39.1-py3-none-win_amd64.whl
```

### Fine-tuning with a Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_chatglm_model \
    --do_train \
    --dataset gender_train \
    --finetuning_type lora \
    --output_dir path_to_sft_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

Please refer to our [Wiki](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/wiki) about the details of the arguments.

### Distributed Fine-tuning with Multiple GPUs

```bash
accelerate config # configure the environment
accelerate launch src/train_bash.py # arguments (same as above)
```

### Evaluation (BLEU and ROUGE_CHINESE)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_chatglm_model \
    --do_eval \
    --dataset alpaca_gpt4_en \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_eval_result \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
```

### Predict

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_your_chatglm_model \
    --do_predict \
    --dataset gender_test \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_predict_result \
    --per_device_eval_batch_size 8 \
    --max_samples 100 \
    --predict_with_generate
```

If you want to predict the samples with empty responses, please kindly fill the `response` column with **dummy tokens** to ensure the sample will not be discarded throughout the preprocessing phase.

### Export model

```bash
python src/export_model.py \
    --model_name_or_path path_to_your_chatglm_model \
    --finetuning_type lora \
    --checkpoint_dir path_to_checkpoint \
    --output_dir path_to_export
```

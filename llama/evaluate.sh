#!/bin/bash
output_dir=./predict/homepage_epoch3
if [ ! -d "$output_dir" ];then
    mkdir -p $output_dir
    echo "创建文件夹成功"
else
    echo "文件夹已经存在"
fi


# # --max_samples 50\
CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_bash.py \
    --stage sft \
    --do_predict \
    --template default\
    --finetuning_type lora \
    --dataset homepage_test\
    --model_name_or_path /home/shishijie/workspace/PTMs/llama-7b-hf \
    --checkpoint_dir /home/shishijie/workspace/project/LLaMA-Efficient-Tuning/output/homepage \
    --output_dir $output_dir \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    >log/evaluate_homepage.log 2>&1 &


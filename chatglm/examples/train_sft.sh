#!/bin/bash
output_dir=../output/diagnosis_train_5e-5_chatglm2
if [ ! -d "$output_dir" ];then
    mkdir -p $output_dir
    echo "创建文件夹成功"
else
    echo "文件夹已经存在"
fi
# --config_file accelerate_config.yaml

CUDA_VISIBLE_DEVICES=0,1,6,7 nohup accelerate launch ../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/shishijie/workspace/PTMs/chatglm2-6b \
    --dataset diagnosis_train \
    --dataset_dir ../data \
    --finetuning_type lora \
    --output_dir $output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_source_length 256 \
    --max_target_length 64 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --fp16 \
    >$output_dir/train.log 2>&1 &
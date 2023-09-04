#!/bin/bash
output_dir=../predict/diagnosis_train_5e-5_chatglm2
if [ ! -d "$output_dir" ];then
    mkdir -p $output_dir
    echo "创建文件夹成功"
else
    echo "文件夹已经存在"
fi

# --checkpoint_dir /home/shishijie/workspace/project/ChatGLM-Efficient-Tuning/output/homepage_epoch3\
# --max_samples 50\
CUDA_VISIBLE_DEVICES=0 python ../src/train_bash.py \
    --stage sft \
    --do_predict \
    --finetuning_type lora \
    --dataset diagnosis_test\
    --dataset_dir ../data \
    --model_name_or_path /home/shishijie/workspace/PTMs/chatglm2-6b \
    --checkpoint_dir /home/shishijie/workspace/project/ChatGLM-Efficient-Tuning/output/diagnosis_train_5e-5_chatglm2\
    --output_dir $output_dir \
    --max_source_length 256 \
    --max_target_length 64 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate

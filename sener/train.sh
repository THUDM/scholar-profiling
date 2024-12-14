#!/bin/bash
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $MASTER_PORT train.py \
    --task scholar-xl \
    --n_epochs 30 \
    --lr 7e-4 \
    --cnn_dim 32 \
    --biaffine_size 100 \
    --chunks_size 128 \
    --batch_size 16 \
    --logit_drop 0.1 \
    --cnn_depth 2

#!/bin/bash

python predict.py --task scholar-xl --cnn_dim 32 --biaffine_size 100 --chunks_size 128 --logit_drop 0.1 --cnn_depth 2

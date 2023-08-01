## CNN-base

### 介绍

该模型是基于[An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition](https://arxiv.org/abs/2208.04534)，源代码是基于fastnlp包，本代码仅复用模型部分，转换为pytorch实现。

### 数据集

数据在“bio_models/en_bio”路径下 数据处理以及切分代码见uie

### 运行

1. 需要修改train_CME.py和predict_CME.py中预训练语言模型以及数据的路径

2. 可调整的参数有b(BATCH_SIZE)、lr、n(EPOCH)、cnn_dim、biaffine_size、n_head、logit_drop、cnn_depth、max_len

#### train

```python
python train_CNN.py -n 30 --lr 7e-6 --cnn_dim 120 --biaffine_size 200 --n_head 5 -b 16 --logit_drop 0.1 --cnn_depth 3
```

#### predict

```
python predict_CNN.py --cnn_dim 120 --biaffine_size 200 --n_head 5 --logit_drop 0.1 --cnn_depth 3
```

### 效果

基于词对匹配，预训练语言模型为bert-base，总的f1值为45.09177(三次随机种子取平均)


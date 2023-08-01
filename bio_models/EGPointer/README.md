## Efficient GlobalPointer

### 介绍

基于 GlobalPointer 的改进，[Keras 版本](https://spaces.ac.cn/archives/8877) 的 torch 复现，核心还是 token-pair 。

### 数据集

数据在“bio_models/en_bio”路径下 数据处理以及切分代码见uie

### 运行

1. 需要修改train_CME.py和predict_CME.py中预训练语言模型以及数据的路径

2. 可调整的参数有BATCH_SIZE、lr、EPOCH、max_len、模型内隐藏层的大小（本代码中为64）

3. GlobalPointer.py中有两个版本的GPointer模型，RawGlobalPointer为原版参数更多，EffiGlobalPointer参数更少，更加高效，性能会略有损失，但整体相差不会很大。可根据需求自行更换。

#### train

```python
python train_CME.py
```

#### predict

```
python predict_CME.py
```

### 效果

采用EffiGlobalPointer，词对匹配方式，预训练语言模型为bert-base，f1值为40.39387(三次随机种子取平均)


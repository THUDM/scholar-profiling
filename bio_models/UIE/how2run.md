## uie

### 介绍

基于[``Unified Structure Generation for Universal Information Extraction``](https://aclanthology.org/2022.acl-long.395/)的实现

### 数据集

数据在“bio_models/en_bio”路径下 

数据处理分为两部分，首先是预处理处理文件为‘UIE/dataset/data_processing.py’，对原始数据进行清洗、划分并将保存为json格式，同时仅保留了部分类别（见处理文件）。数据是按照文件进行切分，防止类别不平衡，训练集：验证集：测试集的比例为1400: 349: 350。

第二步是将上述json格式数据处理为uie模型所需格式，运行代码如下

```python
python uie_convert.py -format spotasoc -config data_config/en_bio -output en_bio
```

### Pretrained Models
You can find the pre-trained models as following CAS Cloud Box/Google Drive links or download models using command `gdown` (`pip install gdown`).

uie-en-base [[CAS Cloud Box]](https://pan.cstcloud.cn/s/w2hTaHYaRWw) [[Google Drive]](https://drive.google.com/file/d/12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-base-en)

uie-en-large [[CAS Cloud Box]](https://pan.cstcloud.cn/s/2vrXYBVTbk) [[Google Drive]](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-large-en)

uie-char-small (chinese) [[CAS Cloud Box]](https://pan.cstcloud.cn/s/J7HOsDHHQHY)

``` bash
# Example of Google Drive
gdown 12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs && unzip uie-base-en.zip
gdown 15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1 && unzip uie-large-en.zip
```

Put all models to `hf_models/` for default running scripts.

### 运行

1. 预训练语言模型(uie-base)需放在“UIE/hf_models”文件夹下

2. 可在‘UIE/config/data_conf/base_model_conf_en_bio.ini’更改训练参数

3. 程序的入口脚本为“UIE/run_uie_finetune.bash”，入口文件为“UIE/run_uie_finetune.py”

```bash
. config/data_conf/base_model_conf_en_bio.ini  && model_name=uie-base-en dataset_name=en_bio/en_bio bash scripts_exp/run_exp.bash
```

### 效果

预训练语言模型为原作者公开的uie-base，string-f1值为36.58, offset-f1值为37.01


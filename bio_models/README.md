This folder provides three profiling methods for extracting more than 10 attributes from long texts of scholars' profiles.

### Implemented Methods
- CNN [1]
- EGPointer [2]
- UIE [3]  

Required pacakges: see `requirements.txt`.

### Data
- Dataset used for CNN and EGPointer is located in `bio_models/en_bio`
- Dataset used for UIE is located in `bio_models/UIE/data/text2spotasoc/en_bio/en_bio`


### Pretrained Models for UIE
You can find the pre-trained models as following CAS Cloud Box/Google Drive links or download models using command `gdown` (`pip install gdown`).

uie-en-base [[CAS Cloud Box]](https://pan.cstcloud.cn/s/w2hTaHYaRWw) [[Google Drive]](https://drive.google.com/file/d/12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-base-en)

uie-en-large [[CAS Cloud Box]](https://pan.cstcloud.cn/s/2vrXYBVTbk) [[Google Drive]](https://drive.google.com/file/d/15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1/view) [[Huggingface]](https://huggingface.co/luyaojie/uie-large-en)

uie-char-small (chinese) [[CAS Cloud Box]](https://pan.cstcloud.cn/s/J7HOsDHHQHY)

Place the uie-base model in `hf_models` folder.


### How to run
```bash
# CNN
cd CNN_base
python train_CNN.py -n 30 --lr 7e-6 --cnn_dim 120 --biaffine_size 200 --n_head 5 -b 16 --logit_drop 0.1 --cnn_depth 2  
python predict_CNN.py --cnn_dim 120 --biaffine_size 200 --n_head 5 --logit_drop 0.1 --cnn_depth 2

# EGPointer
cd EGPointer
python train_CME.py
python predict_CME.py

# UIE
cd UIE
python dataset/data_processing.py
python dataset_processing/uie_convert.py -format spotasoc -config data_config/en_bio -output en_bio
. config/data_conf/base_model_conf_en_bio.ini  && model_name=uie-base-en dataset_name=en_bio/en_bio bash scripts_exp/run_exp.bash

```

### Results
|       | Precision | Recall  |  F1  |
|-------|-------|-----|-----|
| CNN   | 0.4189 | 0.4613 |0.4391 |
| EGpointer | 0.4707 | 0.3770 | 0.4187 |
| UIE   | 0.4240 | 0.3305 | 0.3715 |

### References
[1] Yan, Hang, Yu Sun, Xiaonan Li, and Xipeng Qiu. "An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition." arXiv preprint arXiv:2208.04534 (2022).  
[2] Su, Jianlin, Ahmed Murtadha, Shengfeng Pan, Jing Hou, Jun Sun, Wanwei Huang, Bo Wen, and Yunfeng Liu. "Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition." arXiv preprint arXiv:2208.03054 (2022).  
[3] Lu, Yaojie, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. "Unified Structure Generation for Universal Information Extraction." In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5755-5772. 2022.
  

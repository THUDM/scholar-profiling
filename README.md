# OAG scholar profiling

## Prerequisites

- Linux
- Python 3.7
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/scholar-profiling.git
cd scholar-profiling
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1rpwjKInye7ZptmvkmDTPww) (with password 7lro) or Aliyun. There are three parts as follows:
- data_ex.zip [[Aliyun Link]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/data_ex.zip): unzip the file and put the _data_ directory into project directory.
- pretrain_models.zip [[Aliyun Link]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/pretrain_models.zip): unzip the file and put the _pretrain_models_ directory into project directory.
- googleSearch: use _7z_ to extract data.zip in this folder and put the _googleSearch_ directory in the _data_ directory. [[Aliyun Link1]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z01), [[Aliyun Link2]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z02), [[Aliyun Link3]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z03), [[Aliyun Link4]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z04), [[Aliyun Link5]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z05), [[Aliyun Link6]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z06), [[Aliyun Link7]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z07), [[Aliyun Link8]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.z08), [[Aliyun Link9]](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/scholar-profiling/googleSearch/data.zip)

## How to run
```bash
cd $project_path
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
export PYTHONPATH="`pwd`:$PYTHONPATH"

# Statistical machine learning (SML) methods:
# gender
python sml_baseline/GenderPredict/main.py
# homepage
python sml_baseline/HomepagePrediction/homepage_train.py
# position
python sml_baseline/TitlePrediction/title_main.py
# evaluation
python sml_baseline/merge_results.py
python evaluate.py --hp output/sml/sml_predict_xgboost.json --rf data/raw/ground_truth.json

# BERT
# First, uncomment three functions including create_gender_classification_data(), create_homepage_classification_data(), create_title_classification_data() to generate training data
python bert_baseline/tools.py 
# gender
python bert_baseline/gender_classification_bert.py
# homepage
python bert_baseline/homepage_classification_bert.py
# position
python bert_baseline/title_classification_bert.py
# for evaluation, uncomment merge_result() funciton in bert_baseline/tools.py 
python bert_baseline/tools.py 
python evaluate.py --hp data/luoyang-result_new.json --rf data/raw/ground_truth.json

# Bi-LSTM-CRF for position tagging
python data_process.py
python bert_bilstm_crf/run.py

# BERT with prompt tuning
# First, uncomment four functions including get_gender_data(r'data/raw/new_dev.xlsx'), get_title_data(r'data/raw/new_dev.xlsx'), get_gender_test(), and get_train_data() to generate training data
python data_process.py
# gender
python prompt/gender_prompt.py
# postion
python prompt/title_prompt.py

```

For how to extract more attributes from long texts of scholars' profiles, please see README.md in `bio_models`.


## References
🌟 If you find our work helpful, please leave us a star and cite our paper.
```
@inproceedings{zhang2024oag,
  title={OAG-bench: a human-curated benchmark for academic graph mining},
  author={Fanjin Zhang and Shijie Shi and Yifan Zhu and Bo Chen and Yukuo Cen and Jifan Yu and Yelin Chen and Lulu Wang and Qingfei Zhao and Yuqing Cheng and Tianyi Han and Yuwei An and Dan Zhang and Weng Lam Tam and Kun Cao and Yunhe Pang and Xinyu Guan and Huihui Yuan and Jian Song and Xiaoyan Li and Yuxiao Dong and Jie Tang},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={6214--6225},
  year={2024}
}
```

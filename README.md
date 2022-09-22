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

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1rpwjKInye7ZptmvkmDTPww) (with password 7lro). There are three parts as follows:
- data_ex.zip: unzip the file and put the _data_ directory into project directory.
- pretrain_models.zip: unzip the file and put the _pretrain_models_ directory into project directory.
- googleSearch: use _7z_ to extract data.zip in this folder and put the _googleSearch_ directory in the _data_ directory.

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

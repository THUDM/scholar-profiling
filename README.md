# OAG scholar profiling

## Prerequisites

- Linux
- Python 3

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

The dataset can be downloaded from [BaiduPan]() (with password ). Unzip the file and put the _data_ directory into project directory.

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

```

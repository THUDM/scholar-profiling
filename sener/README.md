<div align="center">

# Small Language Model Makes an Effective Long Text Extractor (AAAI 2025)

</div>


<div align="center">
    <img src=assets/sener.png width=50% />
</div>

*Official code and data of the paper Small Language Model Makes an Effective Long Text Extractor.*

<p align="center">
   📃 <a href="https://arxiv.org/pdf/2502.07286" target="_blank"> Paper </a> 
</p>

***

SeNER, a lightweight span-based NER method with efficient attention mechanisms that enhance long-text encoding and extraction, reduce redundant computations, and achieve state-of-the-art accuracy while being GPU-memory-friendly. 

## 🚀 Quick Start

### Dependencies

First, create a conda environment and install all pip package requirements.

```bash
conda create -n sener python==3.11.9
conda activate sener

cd scholar-profiling/sener
pip install -r requirements.txt
```

### Model checkpoints

The checkpoints has been released here and we use it:

- [scholar-xl checkpoint](https://drive.google.com/file/d/1EZAp5N--a5aWvxL_dGIGnRoG9TSSpyyB/view?usp=sharing)
- [SciREX checkpoint](https://drive.google.com/file/d/1f4GHfMxw0yESEKoz1R2Nr0QAzpbZwFgo/view?usp=sharing)
- [profiling-07 checkpoint](https://drive.google.com/file/d/1w3YiRi_g6UgLPey6CCEgv_fdRZR_P9YH/view?usp=sharing)

### Datasets

The processed data has been released here. We download it and put it in ./data:

- [scholar-xl data](./data/scholar-xl)
- [SciREX data](./data/SciREX)
- [profiling-07 data](./data/profiling-07)

The datasets resources have also been submitted to Hugging Face:

- [scholar-xl data](https://huggingface.co/datasets/QAQ123/Scholar-XL/tree/main)
- [SciREX data](https://huggingface.co/datasets/QAQ123/SciREX/tree/main)
- [profiling-07 data](https://huggingface.co/datasets/QAQ123/Profiling-07/tree/main)

### File structure

You can follow a similar format for processing your data, and then go on to train your own model.

```tree
    - data/
        - scholar-xl
          -train.json
          -dev.json
          -test.json
        - SciREX
        - profiling-07
    - log
    - models/
        - cnn.py
        - metrics.py
        - model.py
    - outputs
    - PLM
    - utils/
        - data_loader.py
        - logger.py
    - predict.py
    - predict.sh
    - train.py
    - train.sh  

```

### ✈️ Train model

We use Deberta-v3-large as the backbone. You can train SeNER with the following commands:

```bash
bash train.sh
```

### 🛜 Evaluation

```bash
bash train.sh
```

## Citation
```
@inproceedings{chen2025small,
  title={Small Language Model Makes an Effective Long Text Extractor},
  author={Chen, Yelin and Zhang, Fanjin and Tang, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

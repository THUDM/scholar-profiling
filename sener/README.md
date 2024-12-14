# SeNER
This is the code for "Small Language Model Makes an Effective Long Text Extractor"

## Requirements
```
python==3.11.9 
fastNLP==1.0.1
transformers==4.37.2                   
pytorch==2.3.1
deepspeed==0.13.4
peft==0.11.1
sentencepiece==0.2.0
```

## File structure

You need to put your data in the parallel folder of this repo.

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
    - utils/
        - data_loader.py
        - logger.py
    - predict.py
    - predict.sh
    - train.py
    - train.sh  

```

## train

```text
sh ./train.sh
```

## predict

```text
sh ./predict.sh
```

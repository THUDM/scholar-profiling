# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

import torch
import torch.nn as nn
import random
import argparse
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

from tqdm import tqdm
from bert_bilstm_crf.dataloader import data_generator
from transformers import get_linear_schedule_with_warmup
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from pathlib import Path

title2Id = {"Other(其他)": 0,
            "Professor(教授)": 1,
            "Researcher(研究员)": 2,
            "Associate Professor(副教授)": 3,
            "Assistant Professor(助理教授)": 4,
            "Professorate Senior Engineer(教授级高级工程师)": 5,
            "Engineer(工程师)": 6,
            "Lecturer(讲师)": 7,
            "Senior Engineer(高级工程师)": 8,
            "Ph.D(博士生)": 9,
            "Associate Researcher(副研究员)": 10,
            "Assistant Researcher(助理研究员)": 11,
            "Student(学生)": 12,
            }

def train_title(args):
    # 加载训练集以及测试集
    train = json.load(open('data/train.json', 'r', encoding='utf-8'))
    # valid = json.load(open('data/title_dev.json', 'r', encoding='utf-8'))
    valid = json.load(open('data/dev.json', 'r', encoding='utf-8'))

    train_dataset = []
    valid_dataset = []
    for data in tqdm(train):
        input_example = InputExample(text_a=data['title_text'], text_b=data['name'], label=title2Id[data['title']],
                                     guid=data['id'])
        train_dataset.append(input_example)
    for data in tqdm(valid):
        input_example = InputExample(text_a=data['text'], text_b=data['name'], label=title2Id[data['title']],
                                     guid=data['id'])
        valid_dataset.append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")  # ("t5", "t5-base")

    promptTemplate = ManualTemplate(tokenizer=tokenizer,
                                    text='{"placeholder":"text_a"} {"placeholder":"text_b"} is {"mask"}.')
    
    
    classes = [  
        "Other(其他)",
        "Professor(教授)",
        "Researcher(研究员)",
        "Associate Professor(副教授)",
        "Assistant Professor(助理教授)",
        "Professorate Senior Engineer(教授级高级工程师)",
        "Engineer(工程师)",
        "Lecturer(讲师)",
        "Senior Engineer(高级工程师)",
        "Ph.D(博士生)",
        "Associate Researcher(副研究员)",
        "Assistant Researcher(助理研究员)",
        "Student(学生)",
    ]
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "Other(其他)": ["Other"],
            "Professor(教授)": ["Professor", "Prof."],
            "Researcher(研究员)": ["Researcher", "Investigator"],
            "Associate Professor(副教授)": ["Assoc. Prof.", "Associate Professor"],
            "Assistant Professor(助理教授)": ["Assistant Professor"],
            "Professorate Senior Engineer(教授级高级工程师)": ["Professorate Senior Engineer"],
            "Engineer(工程师)": ["Engineer"],
            "Lecturer(讲师)": ["Lecturer"],
            "Senior Engineer(高级工程师)": ["Senior Engineer"],
            "Ph.D(博士生)": ["Ph.D. student"],
            "Associate Researcher(副研究员)": ["Associate Researcher"],
            "Assistant Researcher(助理研究员)": ["Assistant Researcher"],
            "Student(学生)": ["Master student", "Bachelor student"],
        },
        tokenizer=tokenizer,
    )

    prompt_model = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    prompt_model.to("cuda")

    train_loader = PromptDataLoader(dataset=train_dataset, template=promptTemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                    batch_size=64, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
    val_loader = PromptDataLoader(dataset=valid_dataset, template=promptTemplate, tokenizer=tokenizer,
                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                  batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                  truncate_method="head")

    output_path = os.path.join("./output/", args.ex_index)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    t_total = len(train_loader) * args.epoch_num  

    no_decay = ["bias", "LayerNorm.weight"]  
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1 = -1.0
    patience_counter = 0
    loss_func = nn.CrossEntropyLoss(reduction="mean")  # none
    
    
    for epoch in range(args.epoch_num):
        prompt_model.train()
        epoch_loss = 0
        print("epoch:%d" % int(epoch))
        
        with tqdm(total=train_loader.__len__(), desc="train", ncols=80) as t:

            for i, batch in enumerate(train_loader):
                train_inputs = batch.to("cuda")
                logits = prompt_model(train_inputs)
                labels = train_inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  
                prompt_model.zero_grad()
                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)
        # torch.save(prompt_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        allpreds = []
        alllabels = []
        prompt_model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(val_loader):
                val_inputs = inputs.to("cuda")
                logits = prompt_model(val_inputs)
                labels = val_inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

        epoch_loss = epoch_loss / train_loader.__len__()
        improve = acc - best_f1
        # stop training based params.patience 
        if improve > 0:
            print("- Found new best F1")
            print("epoch:%d\tloss:%05.3f\tacc:%0.8f\t" % (int(epoch), epoch_loss, acc))
            best_f1 = acc
            torch.save(prompt_model.state_dict(), os.path.join(output_path, "pytorch_model_best.bin"))
            if improve < 1e-5:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            print("epoch:%d\tloss:%05.3f\tacc:%0.8f\t" % (int(epoch), epoch_loss, acc))
            torch.save(prompt_model.state_dict(), os.path.join(output_path, "pytorch_model_last.bin"))
            patience_counter += 1

        # Early stopping 
        if patience_counter > 10 or epoch == args.epoch_num:
            print("Best result: {:0.8f}".format(best_f1))
            break


def evaluate_title(args):
    test_data = json.load(open('data/test.json', 'r', encoding='utf-8'))
    test_dataset = []
    for data in test_data:
        input_example = InputExample(text_a=data['title_text'], text_b=data['name'], label=title2Id[data['title']],
                                     guid=data['id'])
        test_dataset.append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")  # ("t5", "t5-base")

    promptTemplate = ManualTemplate(tokenizer=tokenizer,
                                    text='{"placeholder":"text_a"} {"placeholder":"text_b"} is {"mask"}.')
    classes = [ 
        "Other(其他)",
        "Professor(教授)",
        "Researcher(研究员)",
        "Associate Professor(副教授)",
        "Assistant Professor(助理教授)",
        "Professorate Senior Engineer(教授级高级工程师)",
        "Engineer(工程师)",
        "Lecturer(讲师)",
        "Senior Engineer(高级工程师)",
        "Ph.D(博士生)",
        "Associate Researcher(副研究员)",
        "Assistant Researcher(助理研究员)",
        "Student(学生)",
    ]
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "Other(其他)": ["Other"],
            "Professor(教授)": ["Professor", "Prof."],
            "Researcher(研究员)": ["Researcher", "Investigator"],
            "Associate Professor(副教授)": ["Assoc. Prof.", "Associate Professor"],
            "Assistant Professor(助理教授)": ["Assistant Professor"],
            "Professorate Senior Engineer(教授级高级工程师)": ["Professorate Senior Engineer"],
            "Engineer(工程师)": ["Engineer"],
            "Lecturer(讲师)": ["Lecturer"],
            "Senior Engineer(高级工程师)": ["Senior Engineer"],
            "Ph.D(博士生)": ["Ph.D. student"],
            "Associate Researcher(副研究员)": ["Associate Researcher"],
            "Assistant Researcher(助理研究员)": ["Assistant Researcher"],
            "Student(学生)": ["Master student", "Bachelor student"],
        },
        tokenizer=tokenizer,
    )

    prompt_model = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    prompt_model.to("cuda")

    test_loader = PromptDataLoader(dataset=test_dataset, template=promptTemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                   batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")

    output_path = os.path.join("./output/", args.ex_index)
    prompt_model.load_state_dict(torch.load(os.path.join(output_path, "pytorch_model_best.bin"), map_location="cuda"))

    allpreds = []
    alllabels = []
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            test_inputs = inputs.to("cuda")
            logits = prompt_model(test_inputs)
            labels = test_inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)

    return acc


if __name__ == '__main__':
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization")
    parser.add_argument('--ex_index', type=str, default="title-prompt")
    parser.add_argument('--epoch_num', type=int, default=100, help="number of epochs")
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-5, type=float)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--warmup', default=0.0, type=float)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    train_title(args)
    print(evaluate_title(args))

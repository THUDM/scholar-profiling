# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备

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


def train(args):
    # 加载训练集以及测试集
    train = json.load(open(r'data/train.json', 'r', encoding='utf-8'))
    valid = json.load(open(r'data/gender_dev.json', 'r', encoding='utf-8'))

    train_dataset = []
    valid_dataset = []
    for data in train:
        input_example = InputExample(text_a=data['gender_text'], text_b=data['name'],
                                     label=int(0 if data['gender'] == "female" else 1),
                                     guid=data['id'])
        train_dataset.append(input_example)
    for data in valid:
        input_example = InputExample(text_a=data['text'], text_b=data['name'],
                                     label=int(0 if data['gender'] == "female" else 1),
                                     guid=data['id'])
        valid_dataset.append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")  # ("t5", "t5-base")

    promptTemplate = ManualTemplate(tokenizer=tokenizer,
                                    text='{"placeholder":"text_a"} {"placeholder":"text_b"} is {"mask"}.')
    

    classes = [  
        "male",
        "female"
    ]
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "male": ["man", "male"],
            "female": ["woman", "female"],
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
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                    batch_size=16, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
    val_loader = PromptDataLoader(dataset=valid_dataset, template=promptTemplate, tokenizer=tokenizer,
                                  tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                  batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                  truncate_method="head")

    # output_path = os.path.join("/home/chenyelin/user_profiling/BILSTM_CRF/output/", args.ex_index)
    output_path = os.path.join("output/", args.ex_index)
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

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1 = -1.0
    patience_counter = 0
    loss_func = nn.CrossEntropyLoss(reduction="mean")  # none
    
    # 训练
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


def evaluate(args):
    # f = open('dev.json', 'w', encoding='utf-8')
    test_data = json.load(open(r'data/test.json', 'r', encoding='utf-8'))
    test_dataset = []
    for data in test_data:
        input_example = InputExample(text_a=data['gender_text'], text_b=data['name'],
                                     label=int(0 if data['gender'] == "female" else 1),
                                     guid=data['id'])
        test_dataset.append(input_example)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")  # ("t5", "t5-base")

    promptTemplate = ManualTemplate(tokenizer=tokenizer,
                                    text='{"placeholder":"text_a"} {"placeholder":"text_b"} is {"mask"}.')
    classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
        "male",
        "female"
    ]
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "male": ["man", "male"],
            "female": ["woman", "female"],
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
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3,
                                   batch_size=16, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")

    # output_path = os.path.join("/home/chenyelin/user_profiling/BILSTM_CRF/output/", args.ex_index)
    output_path = os.path.join("output/", args.ex_index)
    prompt_model.load_state_dict(torch.load(os.path.join(output_path, "pytorch_model_best.bin"), map_location="cuda"))

    allpreds = []
    alllabels = []
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            test_inputs = inputs.to("cuda")
            logits = prompt_model(test_inputs)
            labels = test_inputs['label'].cpu()
            labels = np.array(labels)
            
            alllabels.extend(labels.tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    fpr, tpr, th = roc_curve(alllabels, allpreds, pos_label=1)
    print('sklearn', auc(fpr, tpr))

    return acc

def pred(args, model, tokenizer):
    f = open('test_result.json', 'w', encoding='utf-8')
    model.eval()
    test_data = json.load(open(os.path.join(output_path, "test_result.json"), 'r', encoding='utf-8'))
    test_loader = data_generator(args, test_data, tokenizer, args.train_batch_size, random=False, is_train=False)
    per_right, sen_right = 0, 0
    title_pred = {}
    gold_titles = {}  # 按人比较
    gold_title_list = []  # 比较每句
    with tqdm(total=test_loader.__len__(), desc="pred", ncols=80) as t:
        for i, batch in enumerate(test_loader):
            batch_text_tokens = batch[-1]
            ids = batch[0]
            title = batch[3]
            gold_title_list += title
            batch = [torch.tensor(d).to("cuda") for d in batch[1:3]]
            input_ids, attention_mask = batch
            bs, seq_len = input_ids.size()
            # inference
            with torch.no_grad():
                scores, _, pred_titles = model(input_ids, attention_mask)
            for idx in range(bs):
                flag = 1
                if ids[idx] not in gold_titles:
                    gold_titles[ids[idx]] = title[idx]

                if pred_titles[idx] == []:
                    pred_title_texts = []
                else:
                    pred_title_texts = span2str(pred_titles[idx], batch_text_tokens[idx])

                if pred_title_texts == []:
                    if title[idx] == "Other(其他)":
                        sen_right += 1
                        flag = 0
                    if ids[idx] not in title_pred:
                        title_pred[ids[idx]] = ["Other(其他)"]
                        continue
                    elif ids[idx] in title_pred:
                        title_pred[ids[idx]].append("Other(其他)")
                        continue

                pred_title_texts_list = [pred_title_texts[i][0] for i in range(len(pred_title_texts))]
                if max(pred_title_texts_list, key=pred_title_texts_list.count) == title[idx]:
                    sen_right += 1
                    flag = 0
                if ids[idx] not in title_pred:
                    title_pred[ids[idx]] = pred_title_texts_list
                else:
                    title_pred[ids[idx]] += pred_title_texts_list

                if flag == 1:
                    s = json.dumps({
                        'id': ids[idx],
                        'text': _concat(batch_text_tokens[idx]),
                        'gold_title': title[idx],
                        'pred_title': pred_title_texts,
                        'pred_title_list': pred_title_texts_list,
                        'max_pred_title': max(pred_title_texts_list, key=pred_title_texts_list.count),
                    }, ensure_ascii=False, indent=4)
                    f.write(s + '\n')

            t.update(1)
    f.close()
    sen_total = len(gold_title_list)
    per_total = len(title_pred)
    for j, id in enumerate(title_pred.keys()):
        pre_title = max(title_pred[id], key=title_pred[id].count)
        if pre_title == gold_titles[id]:
            per_right += 1
    return per_right / per_total, sen_right / sen_total


if __name__ == '__main__':
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization")
    parser.add_argument('--ex_index', type=str, default="prompt-gender")
    parser.add_argument('--epoch_num', type=int, default=20, help="number of epochs")
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

    train(args)
    print(evaluate(args))

    
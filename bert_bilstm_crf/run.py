# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import torch.nn as nn
from transformers import BertConfig,BertTokenizer
import random

import argparse
import json
import numpy as np


from tqdm import tqdm, trange
from dataloader import data_generator 
from title_bert_bilstm_crf import Bert_LSTM_CRF 
from transformers import get_linear_schedule_with_warmup
from pathlib import Path


title2Id={"Other(其他)": 0,
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


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        if triple[1] >= args.max_len: break
        tage = triple[0]
        title_tokens = tokens[triple[1]:triple[-1]]
        title = _concat(title_tokens)
        output.append((tage, title))
    return output

def _concat(token_list):
    result = ''
    for idx, t in enumerate(token_list):
        if idx == 0:
            result = t
        elif t.startswith('##'):
            result += t.lstrip('##')
        else:
            result += ' ' + t
    return result

def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def train(args):

    
    root_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    bert_model_dir = root_path / 'pretrain_models/bert_base_cased'
    bert_config = BertConfig.from_json_file(os.path.join(bert_model_dir, 'config.json'))
    model = Bert_LSTM_CRF.from_pretrained(config=bert_config, pretrained_model_name_or_path=bert_model_dir)
    tokenizer = BertTokenizer(vocab_file=os.path.join(bert_model_dir, 'vocab.txt'), do_lower_case=True)  # 大小写敏感
    model.to("cuda")

    output_path = os.path.join("./output/",args.ex_index)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    # train_data = json.load(open(r'/home/chenyelin/BILSTM_CRF/title_train.json', 'r', encoding='utf-8'))
    train_data = json.load(open(r'data/tag_title_train.json', 'r', encoding='utf-8'))
    train_loader = data_generator(args, train_data, tokenizer, args.train_batch_size, random=True)


    t_total = len(train_loader) * args.epoch_num  

    no_decay = ["bias", "LayerNorm.weight"]  
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    
    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )

    best_f1= -1.0
    patience_counter = 0
    
    # 训练
    for epoch in range(args.epoch_num):
        model.train()
        epoch_loss = 0
        print("epoch:%d" % int(epoch))
        
        with tqdm(total=train_loader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(train_loader):

                batch = [torch.tensor(d).to("cuda") for d in batch]  
                batch_token_ids, batch_mask, batch_label = batch
                loss = model.neg_log_likelihood_parallel(batch_token_ids, batch_mask, batch_label)
                loss.backward()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                t.update(1)
        # torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

        acc = pred(args, model, tokenizer)
        epoch_loss = epoch_loss / train_loader.__len__()
        improve = acc - best_f1
        # stop training based params.patience 
        if improve > 0:
            print("- Found new best F1")
            print("epoch:%d\tloss:%f\tprecision:%f\t" % (int(epoch), epoch_loss, acc))
            best_f1 = acc
            torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model_best.bin"))
            if improve < 1e-5:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            print("epoch:%d\tloss:%f\tprecision:%f\t" % (int(epoch), epoch_loss, acc))
            torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model_last.bin"))
            patience_counter += 1

        # Early stopping 
        if patience_counter > 10 or epoch == args.epoch_num:
            print("Best result: {:f}".format(best_f1))
            break

        if epoch % 5 == 0:
            print("eval on test...", end=" ")
            print(evaluate(args, model, tokenizer))

def evaluate(args, model, tokenizer):

    model.eval()
    test_data = json.load(open(r'data/tag_title_test.json', 'r', encoding='utf-8'))
    test_loader = data_generator(args, test_data, tokenizer, args.train_batch_size, random=False, is_train=False)
    per_right, sen_right = 0, 0
    title_pred = {}
    gold_titles = {} # 按人比较
    with tqdm(total=test_loader.__len__(), desc="pred", ncols=80) as t:
        for i, batch in enumerate(test_loader):
            batch_text_tokens = batch[-1]
            ids = batch[0]
            title = batch[3]
            batch = [torch.tensor(d).to("cuda") for d in batch[1:3]]
            input_ids, attention_mask = batch
            bs, seq_len = input_ids.size()
            # inference
            with torch.no_grad():
                scores, _, pred_titles = model(input_ids, attention_mask)
            for idx in range(bs):
                # flag = 1
                if ids[idx] not in gold_titles:
                    gold_titles[ids[idx]] = title[idx]

                if pred_titles[idx] == []:
                    pred_title_texts = []
                else:
                    pred_title_texts = span2str(pred_titles[idx], batch_text_tokens[idx])

                if pred_title_texts == []:
                    if ids[idx] not in title_pred:
                        title_pred[ids[idx]] = ["Other(其他)"]
                        continue
                    elif ids[idx] in title_pred:
                        title_pred[ids[idx]].append("Other(其他)")
                        continue

                pred_title_texts_list = [pred_title_texts[i][0] for i in range(len(pred_title_texts))]
                if ids[idx] not in title_pred:
                    title_pred[ids[idx]] = pred_title_texts_list
                else:
                    title_pred[ids[idx]] += pred_title_texts_list

            t.update(1)
   
    per_total = len(title_pred)
    for j, id in enumerate(title_pred.keys()):
        pre_title = max(title_pred[id], key=title_pred[id].count)
        if pre_title == gold_titles[id]:
            per_right += 1
    return per_right / per_total

def pred(args, model, tokenizer):
    
    model.eval()
    test_data = json.load(open('data/tag_title_dev.json', 'r', encoding='utf-8'))
    test_loader = data_generator(args, test_data, tokenizer, args.train_batch_size, random=False, is_train=False)
    per_right = 0
    title_pred = {}
    gold_titles = {} # 按人比较
    with tqdm(total=test_loader.__len__(), desc="pred", ncols=80) as t:
        for i, batch in enumerate(test_loader):
            batch_text_tokens = batch[-1]
            ids = batch[0]
            title = batch[3]
            batch = [torch.tensor(d).to("cuda") for d in batch[1:3]]
            input_ids, attention_mask = batch
            bs, seq_len = input_ids.size()
            # inference
            with torch.no_grad():
                scores, _, pred_titles = model(input_ids, attention_mask)
            for idx in range(bs):
                # flag = 1
                if ids[idx] not in gold_titles:
                    gold_titles[ids[idx]] = title[idx]

                if pred_titles[idx] == []:
                    pred_title_texts = []
                else:
                    pred_title_texts = span2str(pred_titles[idx], batch_text_tokens[idx])

                if pred_title_texts == []:
                    if ids[idx] not in title_pred:
                        title_pred[ids[idx]] = ["Other(其他)"]
                        continue
                    elif ids[idx] in title_pred:
                        title_pred[ids[idx]].append("Other(其他)")
                        continue

                pred_title_texts_list = [pred_title_texts[i][0] for i in range(len(pred_title_texts))]
                if ids[idx] not in title_pred:
                    title_pred[ids[idx]] = pred_title_texts_list
                else:
                    title_pred[ids[idx]] += pred_title_texts_list

            t.update(1)
   
    per_total = len(title_pred)
    for j, id in enumerate(title_pred.keys()):
        pre_title = max(title_pred[id], key=title_pred[id].count)
        if pre_title == gold_titles[id]:
            per_right += 1
    return per_right / per_total



if __name__ == '__main__':
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022, help="random seed for initialization")
    parser.add_argument('--ex_index', type=str, default="bilstm_crf")
    parser.add_argument('--train_batch_size', type=int, default=16, help="train batch size")
    parser.add_argument('--epoch_num', type=int, default=100, help="number of epochs")
    parser.add_argument('--max_len',type=int, default=100, help="最大句子长度")
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--warmup', default=0.0, type=float)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_path = os.path.join("./output/", args.ex_index)
    train(args)
    root_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    bert_model_dir = root_path / 'pretrain_models/bert_base_cased'
    bert_config = BertConfig.from_json_file(os.path.join(bert_model_dir, 'config.json'))
    model = Bert_LSTM_CRF.from_pretrained(config=bert_config, pretrained_model_name_or_path=bert_model_dir)
    tokenizer = BertTokenizer(vocab_file=os.path.join(bert_model_dir, 'vocab.txt'), do_lower_case=True)
    model.to("cuda")
    model.load_state_dict(torch.load(os.path.join(output_path, "pytorch_model_best.bin"), map_location="cuda"))
    print(evaluate(args, model, tokenizer))
 
'''
短句子已过滤
句子中的中文已去除
按句抽取
'''

from utils.data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
import json
from models.model import CNNNer
from models.metrics import MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from utils.bert_optimization import BertAdam
from transformers import set_seed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=4e-5, type=float) # 2e-5
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-n', '--n_epochs', default=10, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=200, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=2022, type=int)


args = parser.parse_args()

bert_model_path = '/shenyelin/bio_baselines/PLM/bert-base-uncased'
train_cme_path = '/shenyelin/bio_baselines/en_bio/en_bio_train.json'
eval_cme_path = '/shenyelin/bio_baselines/en_bio/en_bio_val.json'
device = torch.device("cuda:1")

ENT_CLS_NUM = 12

######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-2
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

set_seed(args.seed)
#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

# train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train , batch_size=args.batch_size, collate_fn=ner_train.collate, shuffle=True, num_workers=16)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl , batch_size=args.batch_size, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)

#GP MODEL
encoder = BertModel.from_pretrained(bert_model_path)
model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                 size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,
                kernel_size=kernel_size, n_head=args.n_head, cnn_depth=args.cnn_depth).to(device)

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

import collections
counter = collections.Counter()
for name, param in model.named_parameters():
    counter[name.split('.')[0]] += torch.numel(param)
print(counter)
print("Total param ", sum(counter.values()))
logger.info(json.dumps(counter, indent=2))
logger.info(sum(counter.values()))

#optimizer
def set_optimizer( model, train_steps=None):
    # 非预训练参数的lr是预训练参数lr的100倍
    for name, param in model.named_parameters():
        name = name.lower()
        if param.requires_grad is False:
            continue
        if 'pretrain_model' in name:
            if 'norm' in name or 'bias' in name:
                ln_params.append(param)
            else:
                non_ln_params.append(param)
        else:
            if 'norm' in name or 'bias' in name:
                non_pretrain_ln_params.append(param)
            else:
                non_pretrain_params.append(param)

    optimizer_grouped_parameters = [
        {'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},
        {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},
        {'params': non_pretrain_ln_params, 'lr': args.lr*non_ptm_lr_ratio, 'weight_decay': 0},
        {'params': non_pretrain_params, 'lr': args.lr*non_ptm_lr_ratio, 'weight_decay': weight_decay}
    ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=0.1,
                         e=1e-8,
                         t_total=train_steps)
    return optimizer

optimizer = set_optimizer(model, train_steps= (int(len(ner_train) / args.batch_size) + 1) * args.n_epochs)

metrics = MetricsCalculator(ent_thres=ent_thres, allow_nested=True)
max_f, max_recall = 0.0, 0.0
for eo in range(args.n_epochs):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, ent_target = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
        loss = model(input_ids, attention_mask, segment_ids, labels)
        loss = loss["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

        avg_loss = total_loss / (idx + 1)
        if idx % 50 == 0:
            logger.info("trian_loss:%f"%(avg_loss))
    
    with torch.no_grad():
        total_X, total_Y, total_Z = [], [], []
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, ent_target = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids, labels)

            f1, p, r = metrics.get_evaluate_fpr(logits["scores"], ent_target, attention_mask)
            total_X.extend(f1)
            total_Y.extend(p)
            total_Z.extend(r)
        
        eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
        f = round(eval_info['f1'],6)
        logger.info('\nEval  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right']))
        for item in entity_info.keys():
            logger.info('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
        torch.save(model.state_dict(), './outputs/TEST_EP_L{}.pth'.format(eo))
        if f > max_f:
            logger.info("find best f1 epoch{}".format(eo))
            torch.save(model.state_dict(), './outputs/TEST_BEST.pth')
            max_f = f
        model.train()
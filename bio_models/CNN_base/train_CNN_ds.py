from utils.data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
import json
from models.model import CNNNer
from models.metrics import MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from transformers import set_seed
import argparse
import deepspeed
from transformers import get_linear_schedule_with_warmup

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=7e-6, type=float) # 2e-5
parser.add_argument('-b', '--batch_size', default=8, type=int)
parser.add_argument('-n', '--n_epochs', default=30, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=120, type=int)
parser.add_argument('--logit_drop', default=0.1, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=2023, type=int)
parser.add_argument('--local_rank', type=int)

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 1,
    "zero_allow_untested_optimizer": True,
    "gradient_clipping": 5,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 7e-6
        }
    },
    "fp16": {
        "enabled": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": False,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients" : True
  }
}

args = parser.parse_args()

# bert_model_path = '/data1/zhangfanjin/cyl/bio_baselines/PLM/bert-base-uncased' # bert_base 路径
train_cme_path = '/data1/zhangfanjin/cyl/bio_baselines/en_bio/new_en_bio_train.json'
eval_cme_path = '/data1/zhangfanjin/cyl/bio_baselines/en_bio/new_en_bio_val.json'

######hyper
ENT_CLS_NUM = 12
non_ptm_lr_ratio = 100
weight_decay = 1e-2
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

set_seed(args.seed)

deepspeed.init_distributed()

# #tokenizer
# tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True) # bert-base
# tokenizer = AutoTokenizer.from_pretrained("/data1/zhangfanjin/cyl/bio_baselines/PLM/roberta-large") # roberta-large
tokenizer = AutoTokenizer.from_pretrained("/data1/zhangfanjin/cyl/bio_baselines/PLM/deberta-v3-large") # deberta-v3-large

#train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer, model_name='deberta')
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer, model_name='deberta')
ner_loader_evl = DataLoader(ner_evl , batch_size=args.batch_size, collate_fn=ner_evl.collate, shuffle=False, num_workers=0)

#GP MODEL
# encoder = BertModel.from_pretrained(bert_model_path) # bert-base
# encoder = AutoModel.from_pretrained("/data1/zhangfanjin/cyl/bio_baselines/PLM/roberta-large") # roberta-large
encoder = AutoModel.from_pretrained("/data1/zhangfanjin/cyl/bio_baselines/PLM/deberta-v3-large") # deberta-v3-large
model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                 size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,
                kernel_size=kernel_size, n_head=args.n_head, cnn_depth=args.cnn_depth).cuda()

# for name, param in model.named_parameters():
#     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)

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
def set_optimizer(model):

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
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    return optimizer

optimizer = set_optimizer(model)
total_steps = (int(len(ner_train) / args.batch_size) + 1) * args.n_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup * total_steps, num_training_steps = total_steps)

model_engine, optimizer, ner_loader_train, _ = deepspeed.initialize(
    config=deepspeed_config,
    model=model,
    training_data=ner_train,
    collate_fn=ner_train.collate,
    optimizer=optimizer,
    lr_scheduler=scheduler)
local_rank = model_engine.local_rank

metrics = MetricsCalculator(ent_thres=ent_thres, allow_nested=True)
max_f = 0.0
for eo in range(args.n_epochs):
    loss_total = 0
    n_item = 0
    
    for idx, batch in enumerate(ner_loader_train):
        
        input_ids, indexes, bpe_len, word_len, matrix, ent_target = batch
        input_ids, bpe_len, indexes, matrix = input_ids.cuda(), bpe_len.cuda(), indexes.cuda(), matrix.cuda()
        loss = model_engine(input_ids, bpe_len, indexes, matrix)
        
        loss = loss["loss"]
        model_engine.backward(loss)
        model_engine.step()
        
        loss_total += loss.item()
        cur_n_item = 8
        n_item += cur_n_item
    
    logger.info(f'*** loss: {loss_total / n_item} ***')

    with torch.no_grad():
        total_X, total_Y, total_Z = [], [], []
        model_engine.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):

            input_ids, indexes, bpe_len, word_len, matrix, ent_target = batch
            input_ids, bpe_len, indexes, matrix = input_ids.cuda(), bpe_len.cuda(), indexes.cuda(), matrix.cuda()
            logits = model_engine(input_ids, bpe_len, indexes, matrix)

            f1, p, r = metrics.get_evaluate_fpr(logits["scores"], ent_target, word_len)
            total_X.extend(f1)
            total_Y.extend(p)
            total_Z.extend(r)

        eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
        f = round(eval_info['f1'],6)
        if local_rank == 0:
            logger.info('\nEval{6}  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right'], eo))
            for item in entity_info.keys():
                logger.info('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
            torch.save(model_engine.state_dict(), './outputs/TEST_EP_L{}.pth'.format(eo))
    
        if f > max_f:
            if local_rank == 0:
                logger.info("find best f1 epoch{}".format(eo))
                torch.save(model_engine.state_dict(), './outputs/TEST_BEST.pth')
            max_f = f
        model_engine.train() 

# deepspeed --include=localhost:4,5 --master_port 12345 train_CNN_ds.py -n 30 --lr 7e-6 --cnn_dim 120 --biaffine_size 200 --n_head 5 -b 8 --logit_drop 0.1 --cnn_depth 3
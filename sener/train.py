from utils.data_loader import EntDataset, load_data
from transformers import AutoTokenizer, AutoModel
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
from modeling_deberta import DebertaModel
from peft import LoraConfig, get_peft_model
import gc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""

    print(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")

def main(args, seed, max_len = 512):

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 1,
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

    if args.task == "scholar-xl":
        train_cme_path = "./data/scholar-xl/train.json"
        eval_cme_path = "./data/scholar-xl/dev.json"
        ENT_CLS_NUM = 12
        lora_lr = 4e-4
        ent2id = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}
    elif args.task == "SciREX":
        train_cme_path = "./data/SciREX/train.json"
        eval_cme_path = "./data/SciREX/dev.json"
        ENT_CLS_NUM = 4
        lora_lr = 3e-4
        ent2id = {"Method": 0, "Task": 1, "Material":2, "Metric": 3}
    elif args.task == "profiling-07":
        train_cme_path = "./data/profiling-07/train.json"
        eval_cme_path = "./data/profiling-07/dev.json"
        ENT_CLS_NUM = 13
        lora_lr = 2e-4
        ent2id = {"interests": 0, "degree": 1, "address":2, "affiliation": 3, "date":4, "major":5, "univ":6, "email":7, "fax":8, "phone":9, "position":10, "contactinfo":11, "education":12}

    id2ent = {}
    for k, v in ent2id.items(): id2ent[v] = k

    weight_decay = 1e-2
    ent_thres = 0.5

    set_seed(seed)

    deepspeed.init_distributed()
    tokenizer = AutoTokenizer.from_pretrained("/workspace/yelin/bio_baselines/PLM/deberta-v3-large")

    ner_train = EntDataset(train_cme_path, tokenizer=tokenizer, ent2id=ent2id, model_name='deberta', max_len=max_len, window=args.chunks_size)
    ner_evl = EntDataset(eval_cme_path, tokenizer=tokenizer, ent2id=ent2id, model_name='deberta', max_len=max_len, window=args.chunks_size, is_train=False)
    ner_loader_evl = DataLoader(ner_evl, batch_size=2, collate_fn=ner_evl.collate, shuffle=False, num_workers=0)
    evl_example = load_data(eval_cme_path, ent2id)

    encoder = DebertaModel.from_pretrained("/workspace/yelin/bio_baselines/PLM/deberta-v3-large")
    model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                    size_embed_dim=0, logit_drop=args.logit_drop,
                   chunks_size=args.chunks_size, cnn_depth=args.cnn_depth, attn_dropout=0.2).cuda()
    
    config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0,
            bias="lora_only",
        )
    model = get_peft_model(model, config)
    # enable trainable params
    for n, p in model.named_parameters():
        if 'pretrain_model' not in n:
            p.requires_grad_()

    model.print_trainable_parameters()

    # optimizer
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

    def set_optimizer(model):

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
            {'params': non_ln_params, 'lr': lora_lr, 'weight_decay': weight_decay},
            {'params': ln_params, 'lr': lora_lr, 'weight_decay': 0},
            {'params': non_pretrain_ln_params, 'lr': args.lr, 'weight_decay': 0},
            {'params': non_pretrain_params, 'lr': args.lr, 'weight_decay': weight_decay},
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

    metrics = MetricsCalculator(ent_thres=ent_thres, id2ent=id2ent, allow_nested=True)
    max_f, max_recall = 0.0, 0.0

    # patience stage
    patience_counter = 0

    for eo in range(args.n_epochs):
        loss_total = 0
        n_item = 0
        for idx, batch in enumerate(ner_loader_train):

            input_ids, indexes, bpe_len, matrix = batch
            input_ids, bpe_len, indexes, matrix = input_ids.cuda(), bpe_len.cuda(), indexes.cuda(), matrix.cuda()
            loss = model_engine(input_ids, bpe_len, indexes, matrix)
            
            loss = loss["loss"]
            model_engine.backward(loss)
            model_engine.step()
            
            loss_total += loss.item()
            cur_n_item = input_ids.shape[0]
            n_item += cur_n_item
        
        if local_rank == 0:
            logger.info(f'*** loss: {loss_total / n_item} ***')
        with torch.no_grad():
            total_X, total_Y, total_Z = [], [], []
            model_engine.eval()
            pre, pre_offset, pre_example_id, word_lens = [], [], [], []
            for batch in tqdm(ner_loader_evl, desc="Valing"):

                input_ids, indexes, bpe_len, word_len, offset, example_id, ent_target = batch
                input_ids, bpe_len, indexes = input_ids.cuda(), bpe_len.cuda(), indexes.cuda()
                logits = model_engine(input_ids, bpe_len, indexes)

                pre += logits["scores"]
                pre_offset += offset
                pre_example_id += example_id
                word_lens += word_len

            total_X, total_Y, total_Z = metrics.get_evaluate_fpr_overlap(evl_example, pre, word_lens, pre_offset, pre_example_id)
            eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
            f = round(eval_info['f1'],6)
            if local_rank == 0:
                logger.info('\nEval{6}  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right'], eo))
                for item in entity_info.keys():
                    logger.info('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
    
            if f > max_f:
                if local_rank == 0:
                    logger.info("find best f1 epoch{}".format(eo))
                    torch.save(model_engine.state_dict(), './outputs/{0}_{1}_{2}.pth'.format(args.task, max_len, seed))
                max_f = f
                patience_counter = 0
            else:
                patience_counter += 1

            model_engine.train()

        if patience_counter >= 10:
            break

if __name__ == '__main__':
    
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=7e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-n', '--n_epochs', default=30, type=int)
    parser.add_argument('--warmup', default=0.1, type=float)
    parser.add_argument('--cnn_depth', default=2, type=int)
    parser.add_argument('--cnn_dim', default=32, type=int)
    parser.add_argument('--logit_drop', default=0.1, type=float)
    parser.add_argument('--biaffine_size', default=100, type=int)
    parser.add_argument('--chunks_size', default=128, type=int)
    parser.add_argument('--task', default="scholar-xl")
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()

    # seed = random.sample(range(1000,10000),3)
    seed = [2288, 3618, 4937]

    max_len = [5120] 

    for l in max_len:
        for idx in range(3):
            main(args, int(seed[idx]), int(l))
            clean_cache()
    
    print("seed", seed)

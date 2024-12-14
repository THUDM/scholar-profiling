from transformers import AutoTokenizer
from models.model import CNNNer
from models.metrics import MetricsCalculator
import torch
from tqdm import  tqdm
import argparse
from utils.data_loader import EntDataset, load_data
from torch.utils.data import DataLoader
from modeling_deberta import DebertaModel
from peft import LoraConfig, get_peft_model

def main(args, seed, mode="test"):
    
    device = torch.device("cuda:4")

    if args.task == "scholar-xl":
        eval_cme_path = "./data/scholar-xl/{}.json".format(mode)
        save_model_path = './outputs/scholar-xl_5120_{}.pth'.format(seed)
        ENT_CLS_NUM = 12
        ent2id = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}
    elif args.task == "SciREX":
        eval_cme_path = "./data/SciREX/{}.json".format(mode)
        save_model_path = './outputs/SciREX_5120_{}.pth'.format(seed)
        ENT_CLS_NUM = 4
        ent2id = {"Method": 0, "Task": 1, "Material":2, "Metric": 3}
    elif args.task == "profiling-07":
        eval_cme_path = "./data/profiling-07/{}.json".format(mode)
        save_model_path = './outputs/profiling-07_5120_{}.pth'.format(seed)
        ENT_CLS_NUM = 13
        ent2id = {"interests": 0, "degree": 1, "address":2, "affiliation": 3, "date":4, "major":5, "univ":6, "email":7, "fax":8, "phone":9, "position":10, "contactinfo":11, "education":12}
    
    id2ent = {}
    for k, v in ent2id.items(): id2ent[v] = k

    BATCH_SIZE = 4

    tokenizer = AutoTokenizer.from_pretrained("/workspace/yelin/bio_baselines/PLM/deberta-v3-large")
    encoder = DebertaModel.from_pretrained("/workspace/yelin/bio_baselines/PLM/deberta-v3-large")
    model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                    size_embed_dim=0, logit_drop=args.logit_drop,
                    chunks_size=args.chunks_size, cnn_depth=args.cnn_depth, attn_dropout=0.2).to(device)
    
    config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query_proj", "value_proj"],
            lora_dropout=0,
            bias="lora_only",
        )
    model = get_peft_model(model, config)

    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_model_path).items()})
    model.eval()

    metrics = MetricsCalculator(ent_thres=0.5, id2ent=id2ent, allow_nested=True)
    ner_test = EntDataset(eval_cme_path, tokenizer=tokenizer, ent2id=ent2id, model_name='deberta', max_len=5120, window=args.chunks_size, is_train=False)
    ner_loader_test = DataLoader(ner_test, batch_size=BATCH_SIZE, collate_fn=ner_test.collate, shuffle=False, num_workers=0)
    evl_example = load_data(eval_cme_path, ent2id)

    with torch.no_grad():
        total_X, total_Y, total_Z = [], [], []
        pre, pre_offset, pre_example_id, word_lens = [], [], [], []
        for batch in tqdm(ner_loader_test, desc="Testing"):

            input_ids, indexes, bpe_len, word_len, offset, example_id, ent_target = batch
            input_ids, bpe_len, indexes = input_ids.to(device), bpe_len.to(device), indexes.to(device) # 
            logits = model(input_ids, bpe_len, indexes)

            pre += logits["scores"]
            pre_offset += offset
            pre_example_id += example_id
            word_lens += word_len
    
    total_X, total_Y, total_Z = metrics.get_evaluate_fpr_overlap(evl_example, pre, word_lens, pre_offset, pre_example_id)       
    eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
    print('\nEval  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right']))
    for item in entity_info.keys():
        print('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_depth', default=2, type=int)
    parser.add_argument('--cnn_dim', default=32, type=int)
    parser.add_argument('--logit_drop', default=0.1, type=float)
    parser.add_argument('--biaffine_size', default=100, type=int)
    parser.add_argument('--chunks_size', default=128, type=int)
    parser.add_argument('--task', default="scholar-xl")

    args = parser.parse_args()

    seed = [2288, 3618, 4937]

    for idx in seed:
        main(args, int(idx))

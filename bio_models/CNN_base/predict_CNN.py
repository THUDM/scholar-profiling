from transformers import BertModel, BertTokenizerFast
from models.model import CNNNer
from models.metrics import MetricsCalculator
import torch
from tqdm import  tqdm
import argparse
from models.metrics import MetricsCalculator
from transformers import set_seed
from utils.data_loader import EntDataset, load_data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--cnn_depth', default=3, type=int)
parser.add_argument('--cnn_dim', default=200, type=int)
parser.add_argument('--logit_drop', default=0, type=float)
parser.add_argument('--biaffine_size', default=200, type=int)
parser.add_argument('--n_head', default=5, type=int)
parser.add_argument('--seed', default=2022, type=int)


args = parser.parse_args()


bert_model_path = 'bert-base-uncased'
save_model_path = './outputs/TEST_BEST.pth'
device = torch.device("cuda:0")

BATCH_SIZE = 16
ENT_CLS_NUM = 12

######hyper
non_ptm_lr_ratio = 100
schedule = 'linear'
weight_decay = 1e-2
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
######hyper

max_len = 512
ent2id, id2ent = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                 size_embed_dim=size_embed_dim, logit_drop=args.logit_drop,
                kernel_size=kernel_size, n_head=args.n_head, cnn_depth=args.cnn_depth).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

set_seed(2022)

metrics = MetricsCalculator(ent_thres=ent_thres, allow_nested=True)

ner_test = EntDataset(load_data('../en_bio/en_bio_test.json'), tokenizer=tokenizer)
ner_loader_test = DataLoader(ner_test , batch_size=BATCH_SIZE, collate_fn=ner_test.collate, shuffle=False, num_workers=16)
with torch.no_grad():
    total_X, total_Y, total_Z = [], [], []
    for batch in tqdm(ner_loader_test, desc="Testing"):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, ent_target = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids, labels)

        f1, p, r = metrics.get_evaluate_fpr(logits["scores"], ent_target, attention_mask)
        total_X.extend(f1)
        total_Y.extend(p)
        total_Z.extend(r)

eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
print('\nEval  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right']))
for item in entity_info.keys():
    print('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))

from transformers import BertModel, BertTokenizerFast
from models.GlobalPointer import EffiGlobalPointer as GlobalPointer
import torch
from tqdm import  tqdm
from transformers import set_seed
from utils.data_loader import EntDataset, load_data
from torch.utils.data import DataLoader
from models.GlobalPointer import MetricsCalculator

bert_model_path = '/shenyelin/bio_baselines/PLM/bert-base-uncased' # 模型路径

save_model_path = './outputs/TEST_BEST.pth'
device = torch.device("cuda:1")

max_len = 512 # 256
ent2id, id2ent = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder =BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, 12 , 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:1'))
model.eval()

BATCH_SIZE = 16
ENT_CLS_NUM = 12 # 实体类型数量

set_seed(2022)

metrics = MetricsCalculator()

ner_test = EntDataset(load_data('/shenyelin/bio_baselines/en_bio/en_bio_test.json'), tokenizer=tokenizer)
ner_loader_test = DataLoader(ner_test , batch_size=BATCH_SIZE, collate_fn=ner_test.collate, shuffle=False, num_workers=16)

with torch.no_grad():
    total_X, total_Y, total_Z = [], [], []
    for batch in tqdm(ner_loader_test, desc="Testing"):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, ent_target = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
            device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids).data.cpu().numpy()

        f1, p, r = metrics.get_evaluate_fpr(logits, ent_target)
        total_X.extend(f1)
        total_Y.extend(p)
        total_Z.extend(r)

eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
print('\nEval  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right']))
for item in entity_info.keys():
    print('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
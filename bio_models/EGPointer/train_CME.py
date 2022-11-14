from utils.data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
import torch
from models.GlobalPointer import EffiGlobalPointer, MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from utils.bert_optimization import BertAdam
from transformers import set_seed

bert_model_path = 'bert-base-uncased' # 模型路径
train_cme_path = '../en_bio/en_bio_train.json'  #CMeEE 训练集
eval_cme_path = '../en_bio/en_bio_val.json'  #CMeEE 测试集
device = torch.device("cuda:0")

BATCH_SIZE = 16
ENT_CLS_NUM = 12 # 分类数量

set_seed(2022)
#tokenizer
tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

# train_data and val_data
ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
ner_loader_train = DataLoader(ner_train , batch_size=BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True, num_workers=16)
ner_evl = EntDataset(load_data(eval_cme_path), tokenizer=tokenizer)
ner_loader_evl = DataLoader(ner_evl , batch_size=BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False, num_workers=16)

#GP MODEL
encoder = BertModel.from_pretrained(bert_model_path)
model = EffiGlobalPointer(encoder, ENT_CLS_NUM, 64).to(device) # 12个实体类型

#optimizer
def set_optimizer( model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=4e-5, # 2e-5
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer
EPOCH = 30 # 10
optimizer = set_optimizer(model, train_steps= (int(len(ner_train) / BATCH_SIZE) + 1) * EPOCH)

# 根据attentionmask处理mask
def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1]) # 添加一个额外的0类
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(EPOCH):
    total_loss, total_f1 = 0., 0.
    for idx, batch in enumerate(ner_loader_train):
        raw_text_list, input_ids, attention_mask, segment_ids, labels, spoes = batch
        input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(device), segment_ids.to(device), labels.to(device)
        logits = model(input_ids, attention_mask, segment_ids)
        loss = loss_fun(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sample_f1 = metrics.get_sample_f1(logits, labels)
        total_loss+=loss.item()
        total_f1 += sample_f1.item()

        avg_loss = total_loss / (idx + 1)
        avg_f1 = total_f1 / (idx + 1)
        if idx % 50 == 0:
            logger.info("trian_loss:%f\t train_f1:%f"%(avg_loss, avg_f1))

    with torch.no_grad():
        total_X, total_Y, total_Z = [], [], []
        model.eval()
        for batch in tqdm(ner_loader_evl, desc="Valing"):
            raw_text_list, input_ids, attention_mask, segment_ids, labels, ent_target = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids).data.cpu().numpy()

            f1, p, r = metrics.get_evaluate_fpr(logits, ent_target)
            total_X.extend(f1)
            total_Y.extend(p)
            total_Z.extend(r)
        
        eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
        f = round(eval_info['f1'],6)
        # 打印总的以及每个类别的评价指标
        logger.info('\nEval  precision:{0}  recall:{1}  f1:{2}  origin:{3}  found:{4}  right:{5}'.format(round(eval_info['acc'],6), round(eval_info['recall'],6), round(eval_info['f1'],6), eval_info['origin'], eval_info['found'], eval_info['right']))
        for item in entity_info.keys():
            logger.info('-- item:  {0}  precision:{1}  recall:{2}  f1:{3}  origin:{4}  found:{5}  right:{6}'.format(item, round(entity_info[item]['acc'],6), round(entity_info[item]['recall'],6), round(entity_info[item]['f1'],6), entity_info[item]['origin'], entity_info[item]['found'], entity_info[item]['right']))
        torch.save(model.state_dict(), './outputs/TEST_EP_L{}.pth'.format(eo))
        if f > max_f:
            logger.info("find best f1 epoch{}".format(eo))
            torch.save(model.state_dict(), './outputs/TEST_BEST.pth')
            max_f = f
        model.train()

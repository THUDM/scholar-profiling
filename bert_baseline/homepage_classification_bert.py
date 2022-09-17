import os
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import time
# import tensorflow as tf
# from keras.backend import  tensorflow_backend as K
import random
import tools

import settings

set_gelu('tanh')  # 切换gelu版本

num_classes = 2
maxlen = 128
batch_size = 8
# bert配置
# path = r'/DATA/disk1/model_data/wll_data/kaiyu/172.20.2.2/kaiyu/profiling/uncased_L-12_H-768_A-12'
# path = "/home/zfj/research-data/user_profiling/bert_model"
path = os.path.join(settings.DATA_DIR, "bert_model")
# path = r'C:\Users\PC\PycharmProjects\BERT\data\Bert模型\cased_L-12_H-768_A-12'
config_path = path + '/bert_config.json'
checkpoint_path = path + '/bert_model.ckpt'
dict_path = path + '/vocab.txt'
bert_input_dir = os.path.join(settings.DATA_DIR, "bert_input")


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if l.strip()=='':continue
            try:
                text, label = l.strip().split('\t')
                name,url =text.split('[SEP]')
                domain = tools.get_domain(url)
                if domain in tools.neg_domains:
                    if random.random()<0.1:
                        D.append((text, int(label)))
                else:
                    D.append((text, int(label)))
            except:
                print(l)
    return D


# data = load_data(r'homepage.train')
data = load_data(os.path.join(bert_input_dir, "homepage.train"))
print('positive:',len([d for d in data if d[1]==1]))
print('negetive:',len([d for d in data if d[1]==0]))
k = int(len(data)*0.9)
valid_data = data[k:]
train_data = data[:k]
print("train", len(train_data), "valid", len(valid_data))
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

max_epoch = 40


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units = num_classes,
    activation = 'softmax',
    kernel_initializer = bert.initializer
)(output)
model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(lr=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

from tqdm import tqdm
def evaluate(data):
    pbar = tqdm()
    X = 1e-14
    Y = 1e-14
    Z = 1e-14
    f1, precision, recall =0,0,0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        R =  set(np.where(y_pred[:]==1)[0].tolist())
        T =  set(np.where(y_true[:]==1)[0].tolist())
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
    pbar.close()
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        self.best_f1 = 0.
        self.stopCount = 0  # 在验证集上停止增长的次数
        self.count = 0
        #super(EarlyStoppingAtMinLoss, self).__init__()
        # 耐心度与最佳权重保存
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall  = evaluate(valid_generator)
        if f1 > self.best_f1:
            self.best_f1 = f1
            model.save_weights('homepage_best_model.weights')

        print(
            u'f1: %.5f,precision: %.5f,recall: %.5f, best_f1: %.5f\n' %
            (f1, precision,recall,self.best_f1)
        )

# def evaluate(data):
#     total, right = 0., 0.
#     for x_true, y_true in data:
#         y_pred = model.predict(x_true).argmax(axis=1)
#         y_true = y_true[:, 0]
#         total += len(y_true)
#         right += (y_true == y_pred).sum()
#     return right / total

# class Evaluator(keras.callbacks.Callback):
#     """Stop training when the loss is at its min, i.e. the loss stops decreasing.
#   Arguments:
#       patience: Number of epochs to wait after min has been hit. After this
#       number of no improvement, training stops.
#   """
#
#     def __init__(self, patience=0):
#         self.best_val_acc = 0.
#         self.stopCount = 0  # 在验证集上停止增长的次数
#         self.count = 0
#         #super(EarlyStoppingAtMinLoss, self).__init__()
#         # 耐心度与最佳权重保存
#         self.patience = patience
#
#     def on_epoch_end(self, epoch, logs=None):
#         val_acc = evaluate(valid_generator)
#         if val_acc > self.best_val_acc:
#             self.best_val_acc = val_acc
#             model.save_weights('homepage_best_model.weights')
#
#         print(
#             u'val_acc: %.5f, best_val_acc: %.5f\n' %
#             (val_acc, self.best_val_acc)
#         )

# 转换数据集
def train():
    train_generator = data_generator(train_data, batch_size)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=max_epoch,
        verbose=1,
        callbacks=[Evaluator()]
    )

from tools import  get_search_list,r_excel_list
import tools
import json



def pred(file):
    data = tools.r_excel_list(file)
    outf =open(file.replace('.xlsx','-homepage.json'),'w',encoding='utf-8')
    for d in data:
        id = d['id']
        name = d['name']
        org = d['org']
        homepages = []
        link_list = get_search_list(id)
        if  link_list:
            for link in link_list:
                href = link['href']
                url = tools.clear_url(href)
                text =  name+'[SEP]' + url
                domain = tools.get_domain(url)
                if  domain in tools.neg_domains:
                    pass
                else:
                    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                    pred_list=model.predict([[token_ids], [segment_ids]])
                    y_pred =np.where(pred_list > 0.9, 1, 0).argmax(axis=1)[0]
                    if y_pred==1 :homepages.append(href)

        s = json.dumps({
            'id': id,
            'name':name,
            'org':org,
            'homepage': homepages
        },
            ensure_ascii=False)
        outf.write(s + '\n')
    outf.close()

# model.load_weights('homepage_best_model.weights')
train()
evaluate(valid_generator)
# test_path="../../data/new_test.xlsx"
test_path = os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx")
pred(test_path)

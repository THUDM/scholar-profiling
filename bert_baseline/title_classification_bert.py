import os
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import time
import tensorflow as tf
from keras.backend import  tensorflow_backend as K
import random

import settings

set_gelu('tanh')  # 切换gelu版本

num_classes = 14
maxlen = 512
batch_size = 8
# bert配置
# path = r'C:\Users\PC\PycharmProjects\BERT\data\Bert模型\chinese_L-12_H-768_A-12'
# path = r'/DATA/disk1/model_data/wll_data/kaiyu/172.20.2.2/kaiyu/profiling/uncased_L-12_H-768_A-12'
# path = "/home/zfj/research-data/user_profiling/bert_model"
# path = r'C:\Users\PC\PycharmProjects\BERT\data\Bert模型\cased_L-12_H-768_A-12'
path = os.path.join(settings.DATA_DIR, "bert_model")
config_path = path + '/bert_config.json'
checkpoint_path = path + '/bert_model.ckpt'
dict_path = path + '/vocab.txt'
bert_input_dir = os.path.join(settings.DATA_DIR, "bert_input")

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            print(l, l.strip().split('\t')[-1])
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# data = load_data(r'title.train')
data = load_data(os.path.join(bert_input_dir, "title.train"))
valid_data = data[5000:]
train_data = data[:5000]

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

max_epoch = 20


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=False,
    # hierarchical_position=True
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
    optimizer=AdamLR(lr=1e-5, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
best_epoch = None

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.
  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        self.best_val_acc = 0.
        self.stopCount = 0  # 在验证集上停止增长的次数
        self.count = 0
        #super(EarlyStoppingAtMinLoss, self).__init__()
        # 耐心度与最佳权重保存
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            best_epoch = epoch
            self.best_val_acc = val_acc
            os.makedirs("output/bert/", exist_ok=True)
            # model.save_weights('title_best_model.weights')
            model.save_weights("output/bert/title_best_model.weights")

        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

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
# 转换数据集
from tools import  get_search_list
import tools
import json
id2title ={0: 'Other(其他)', 1: 'Professor(教授)', 2: 'Researcher(研究员)', 3: 'Associate Professor(副教授)', 4: 'Assistant Professor(助理教授)', 5: 'Professorate Senior Engineer(教授级高级工程师)', 6: 'Engineer(工程师)', 7: 'Lecturer(讲师)', 8: 'Senior Engineer(高级工程师)', 9: 'Ph.D(博士生)', 10: 'Associate Researcher(副研究员)', 11: 'Assistant Researcher(助理研究员)', 12: 'Research(研究员)', 13: 'Student(学生)'}


def get_title_count(text): #效果不好
    title_count={}
    title_count['Associate Professor(副教授)'] = text.count('associate professor')
    title_count['Assistant Professor(助理教授)'] = text.count('assistant professor')
    title_count['Professor(教授)'] = text.count('professor') - title_count['Associate Professor(副教授)'] -title_count['Assistant Professor(助理教授)']

    title_count['Associate Researcher(副研究员)'] = text.count('associate researcher')
    title_count['Assistant Researcher(助理研究员)'] = text.count('assistant researcher')
    title_count['Researcher(研究员)'] = text.count('researcher') -  title_count['Associate Researcher(副研究员)'] -title_count['Assistant Researcher(助理研究员)']
    title_count['Lecturer(讲师)'] = text.count('lecturer')
    title_count['Ph.D(博士生)'] = text.count('ph.d')

    title_count['Professorate Senior Engineer(教授级高级工程师)'] = text.count('professorate senior engineer ')
    title_count['Senior Engineer(高级工程师)'] = text.count('senior engineer')
    title_count['Engineer(工程师)'] = text.count('engineer ') -title_count['Professorate Senior Engineer(教授级高级工程师)'] - title_count['Senior Engineer(高级工程师)']


    items = title_count.items()
    items = list(items)
    items.sort(key=lambda x:x[1],reverse=True)
    print(items)
    if items[0][1]>0:
        return items[0][0]
    else:
        return ''

def pred_text(text):
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    y_pred = model.predict([[token_ids], [segment_ids]]).argmax(axis=1)[0]
    title = id2title[y_pred]
    return title
#直接使用神网络效果最好，最佳正确率为84%

def contain_words(text,words): #判断文本中是否包含一组词语的至少一个
    text = text.lower()
    for word in words:
        word = word.lower()
        if text.find(word)>=0:
            return text
    return ''

def pred(file):
    data = tools.r_excel_list(file)
    outf =open(file.replace('.xlsx','-title.json'),'w',encoding='utf-8')
    words = { 'Professor', 'Researcher', 'Engineer ','Lecturer', 'Ph.D', 'Research ', 'Student'}
    for d in data:
        id = d['id']
        name = d['name']
        org = d['org']
        link_list = get_search_list(id)
        text = name + '; ' + org + '; '
        if link_list:
            for link in link_list:
                if link['content']:
                    word = contain_words(link['content'], words)
                    text += word + '; '
        text = text.replace('\n', ' ').replace('\t', ' ')
        title = pred_text(text)
        s = json.dumps({
            'id': id,
            'name':name,
            'org':org,
            'title': title
        },
            ensure_ascii=False)
        outf.write(s + '\n')
    outf.close()
    
train()

print("best epoch", best_epoch)
model.load_weights('output/bert/title_best_model.weights')

print(evaluate(valid_generator))
# test_path="../../data/new_test.xlsx"
test_path = os.path.join(settings.DATA_DIR, "raw", "new_test.xlsx")
pred(test_path)
# def valid2(): #
#     count = 0
#     total = 0
#     for  d in valid_data:
#         text,id =d
#         label = id2title[id]
#         # title = get_title_count(text)
#         title =''
#         if title=='':
#             title = pred_text(text)
#         if title == label:
#             count+=1
#         # if label =='Professor(教授)':
#         #     count+=1
#     print(count/len(valid_data))
# valid2()
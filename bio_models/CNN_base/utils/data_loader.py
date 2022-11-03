import json
import torch
from torch.utils.data import Dataset
import numpy as np

max_len = 512
ent2id = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}
id2ent = {}
for k, v in ent2id.items(): id2ent[v] = k

def search(pattern, sequence, idx=1):
    """从sequence中寻找第idx个子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    f = 0
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            f += 1
            if f == idx:
                return i
    return -1

def load_data(path):
    D = []
    for data in open(path):
        d = json.loads(data)
        D.append([d['text']])
        for e in d['ner']:
            D[-1].append((e[0], ent2id[e[-1]]))
    return D

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            encoder_txt = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, input_ids, token_type_ids, attention_mask 
        else:
            #TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        spoes_list = []
        for item in examples:
            raw_text, input_ids, token_type_ids, attention_mask = self.encoder(item) 
            text_ids = self.tokenizer.encode(raw_text)
            labels = np.zeros((max_len, max_len, len(ent2id)))
            spoes = set() # 会筛去三个重复项 暂未找出原因
            en = {}
            for entity, label in item[1:]:
                if entity not in en:
                    en[entity] = 1
                else: en[entity] +=1
                o = self.tokenizer.encode(entity, add_special_tokens=False)
                oh = search(o, input_ids, en[entity])
                # oh = search(o, input_ids)
                if oh != -1:
                    labels[oh, oh+len(o)-1, label] = 1
                    labels[oh+len(o)-1, oh, label] = 1
                oh = search(o, text_ids, en[entity])
                # oh = search(o, text_ids)
                if oh != -1:
                    spoes.add((oh, oh+len(o)-1, label))

            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:len(input_ids), :len(input_ids), :])
            spoes_list.append(spoes)

        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=2)).long()
        return raw_text_list, batch_inputids, batch_attentionmask, batch_segmentids, batch_labels, spoes_list

    def __getitem__(self, index):
        item = self.data[index]
        return item


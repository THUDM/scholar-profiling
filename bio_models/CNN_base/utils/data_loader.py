import json
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
# import sparse

ent2id = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}
id2ent = {}
for k, v in ent2id.items(): id2ent[v] = k


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """
    
    def __init__(self,
                 input_ids,
                 indexes,
                 bpe_len,
                 word_len,
                 matrix,
                 ent_target
                 ):
        self.input_ids = input_ids
        self.indexes = indexes
        self.bpe_len = bpe_len
        self.word_len = word_len
        self.matrix = matrix
        self.ent_target = ent_target

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
    num_l = 0
    for data in open(path):
        d = json.loads(data)
        D.append([d['text']])
        for e in d["entity"]:
            num_l +=1 
            start, end, entity, label = e
            if start <= end:
                D[-1].append((start, end-1, ent2id[label]))
    print(num_l)
    return D

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, model_name='bert', max_len=512, istrain=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.istrain = istrain
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            raise RuntimeError(f"Unsupported {model_name}")
        self.data = self.convert(data)

    def __len__(self):
        return len(self.data)
    
    def get_new_ins(self, bpes, spans, indexes):
            if len(bpes)<self.max_len:
                bpes.append(self.sep)
                cur_word_idx = indexes[-1]
                indexes.append(0)
                # int8范围-128~127
                matrix = np.zeros((cur_word_idx, cur_word_idx, len(ent2id)), dtype=np.int8)
                ent_target = []
                for _ner in spans:
                    s, e, t = _ner
                    matrix[s, e, t] = 1
                    matrix[e, s, t] = 1
                    ent_target.append((s, e, t))
                # matrix = sparse.COO.from_numpy(matrix)
            
            else:
                bpes = bpes[:self.max_len]
                indexes = indexes[:self.max_len]
                cur_word_idx = indexes[-1]
                # int8范围-128~127
                matrix = np.zeros((cur_word_idx, cur_word_idx, len(ent2id)), dtype=np.int8)
                ent_target = []
                for _ner in spans:
                    s, e, t = _ner
                    if s < cur_word_idx and e < cur_word_idx:
                        matrix[s, e, t] = 1
                        matrix[e, s, t] = 1
                    ent_target.append((s, e, t))

            # assert len(bpes)<=512, len(bpes)
            new_ins = InputFeatures(input_ids=bpes, indexes=indexes, bpe_len=len(bpes),
                               word_len=cur_word_idx, matrix=matrix, ent_target=ent_target)
            
            return new_ins

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

    def convert(self, data):
        ins_lst = []
        word2bpes = {}
        for item in data:
            raw_words = item[0]
            raw_ents = item[1:]
            old_ent_str = Counter()
            has_ent_mask = np.zeros(len(raw_words), dtype=bool)
            for s, e, t in raw_ents:
                old_ent_str[''.join(raw_words[s:e+1])] += 1
                has_ent_mask[s:e+1] = 1
            punct_indexes = []
            for idx, word in enumerate(raw_words):
                # is_upper = True
                # if idx<len(raw_words):
                #     is_upper = raw_words[idx][0].isupper()
                if self.istrain:
                    if word[-1] == '.' and has_ent_mask[idx] == 0:  # truncate too long sentence.
                        punct_indexes.append(idx+1)

            if len(punct_indexes) == 0 or punct_indexes[-1] != len(raw_words):
                punct_indexes.append(len(raw_words))

            raw_sents = []
            raw_entss = []
            last_end_idx = 0
            for p_i in punct_indexes:
                raw_sents.append(raw_words[last_end_idx:p_i])
                cur_ents = [(s-last_end_idx, e-last_end_idx, t) for s, e, t in raw_ents if last_end_idx<=s<=e<p_i]
                raw_entss.append(cur_ents)
                last_end_idx = p_i

            bpes = [self.cls]
            indexes = [0]
            spans = []
            new_ent_str = Counter()
            for _raw_words, _raw_ents in zip(raw_sents, raw_entss):
                _indexes = []
                _bpes = []
                for s, e, t in _raw_ents:
                    new_ent_str[''.join(_raw_words[s:e+1])] += 1

                for idx, word in enumerate(_raw_words, start=0):
                    if word in word2bpes:
                        __bpes = word2bpes[word]
                    else:
                        __bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                       add_special_tokens=False)
                        word2bpes[word] = __bpes
                    _indexes.extend([idx]*len(__bpes))
                    _bpes.extend(__bpes)
                next_word_idx = indexes[-1]+1
                if len(bpes) + len(_bpes) <= self.max_len:
                    bpes = bpes + _bpes
                    indexes += [i + next_word_idx for i in _indexes]
                    spans += [(s+next_word_idx-1, e+next_word_idx-1, t) for s, e, t in _raw_ents]
                else:
                    new_ins = self.get_new_ins(bpes, spans, indexes)
                    ins_lst.append(new_ins)
                    indexes = [0] + [i + 1 for i in _indexes]
                    spans = [(s, e, t) for s, e, t in _raw_ents]
                    bpes = [self.cls] + _bpes
            if bpes:
                new_ins = self.get_new_ins(bpes, spans, indexes)
                ins_lst.append(new_ins)
            
            assert len(new_ent_str - old_ent_str) == 0 and len(old_ent_str-new_ent_str)==0
        
        return ins_lst
        

    def collate(self, examples):
        batch_input_id, batch_index, batch_bpe_len, batch_word_len, batch_matrix, batch_ent_target = [], [], [], [], [], []

        for item in examples:
            batch_input_id.append(item.input_ids)
            batch_index.append(item.indexes)
            batch_bpe_len.append(item.bpe_len)
            batch_word_len.append(item.word_len)
            batch_matrix.append(item.matrix)
            batch_ent_target.append(item.ent_target)

        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
        batch_indexes = torch.tensor(self.sequence_padding(batch_index)).long()
        batch_bpe_lens = torch.tensor(batch_bpe_len).long()
        batch_word_lens = torch.tensor(batch_word_len).long()
        batch_labels = torch.tensor(self.sequence_padding(batch_matrix, value=-100, seq_dims=2)).long()
        return batch_input_ids, batch_indexes, batch_bpe_lens, batch_word_lens, batch_labels, batch_ent_target

    def __getitem__(self, index):
        item = self.data[index]
        return item

if __name__ == '__main__':

    from transformers import BertTokenizerFast
    from torch.utils.data import DataLoader

    bert_model_path = '/data1/zhangfanjin/cyl/bio_baselines/PLM/bert-base-uncased'
    train_cme_path = '/data1/zhangfanjin/cyl/bio_baselines/en_bio/new_en_bio_train.json'

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, do_lower_case=True)

    ner_train = EntDataset(load_data(train_cme_path), tokenizer=tokenizer)
    # ner_loader_train = DataLoader(ner_train , batch_size=16, collate_fn=ner_train.collate, shuffle=True, num_workers=0)
    print(len(ner_train))
    print(ner_train[7].input_ids)

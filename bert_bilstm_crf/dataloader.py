# /usr/bin/env python
# coding=utf-8
"""Dataloader"""


import numpy as np

class DataGenerator(object):
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        if random:
            if self.steps is None:
                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:
                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        for d in self.__iter__(True):
            yield d

def sequence_padding(inputs,dim=0, length=None, padding=0):
    if not type(inputs[0]) is np.ndarray:
        inputs = [np.array(i) for i in inputs]

    if length is None:
        length = max([x.shape[dim] for x in inputs])
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        pad_width[dim] = (0, length - x.shape[dim])
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)
    return np.array(outputs)

def judge(ex):
    if ex['text'] == [] or ex['text_tags'] == []:
        return False
    else:
        return True

def find_head_idx(source, target):
    head_idx=[]
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            head_idx.append(i)
    return head_idx


class data_generator(DataGenerator):
    def __init__(self,params,train_data, tokenizer, batch_size,random=False,is_train=True):
        super(data_generator,self).__init__(train_data,batch_size)
        self.max_len=params.max_len
        self.tokenizer = tokenizer
        self.random=random
        self.is_train=is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label = []
        batch_title = []
        batch_ids = []
        bach_token = []
        for is_end, d in self.sample(self.random):

            if judge(d)==False:
                continue

            text_tokens = self.tokenizer.tokenize(d['text'])
            text_tags = d['text_tags']
            assert len(text_tokens) == len(d['text_tags'])
            if len(text_tokens) > self.max_len:
                text_tokens = text_tokens[:self.max_len]
                text_tags = text_tags[:self.max_len]
            token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
            mask = [1] * len(token_ids)

            if self.is_train:
                for a, b in zip([batch_token_ids, batch_mask, batch_label],
                                [token_ids, mask, text_tags]):
                    a.append(b)

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    batch_label = sequence_padding(batch_label)
                    yield [
                        batch_token_ids, batch_mask,
                        batch_label
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_label = []
            else:
                for a, b in zip([batch_ids, batch_token_ids, batch_mask, batch_title, batch_label, bach_token],
                                [d['id'], token_ids, mask, d['title'], text_tags, text_tokens]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    batch_label = sequence_padding(batch_label)
                    yield [
                        batch_ids, batch_token_ids, batch_mask, batch_title, batch_label, bach_token
                    ]
                    batch_ids = []
                    batch_token_ids, batch_mask = [], []
                    batch_title = []
                    batch_label = []
                    bach_token = []




if __name__ == '__main__':
    import argparse
    import json
    import os
    from pathlib import Path
    from transformers import BertConfig, BertTokenizer

    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--cuda_id', default="0", type=str)
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--test_batch_size', default=6, type=int)
    parser.add_argument('--num_train_epochs', default=50, type=int)
    parser.add_argument('--max_len', default=100, type=int)

    args = parser.parse_args()

    train_data = json.load(open('train.json'))
    # valid_data = json.load(open(dev_path))
    # test_data = json.load(open(os.path.join(params.data_dir, 'test_triples.json')))
    root_path = Path(os.path.abspath(os.path.dirname(__file__)))
    bert_model_dir = root_path / 'pretrain_models/bert_base_cased'
    tokenizer = BertTokenizer(vocab_file=os.path.join(bert_model_dir, 'vocab.txt'), do_lower_case=True)
    train_loader = data_generator(args, train_data, tokenizer, args.batch_size, random=False,is_train=False)
    # dev_dataloader = data_generator(params, valid_data, [predicate2id, id2predicate],params.val_batch_size, random=False, is_train=False)
    # val_loader = data_generator(params, test_data,rel2idx,params.test_batch_size, random=False, is_train=False)

    for batch in train_loader:
        batch_ids, input_ids, attention_mask, batch_label, batch_text = batch
        print(batch_ids)
        print(batch_label)
        print(batch_text)
        print("------")



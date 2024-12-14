import json
import torch
from torch.utils.data import Dataset
import numpy as np
import random

class InputFeatures(object):
    
    def __init__(self,
                 input_ids,
                 indexes,
                 bpe_len,
                 word_len=None,
                 matrix=None,
                 offset=None,
                 example_id=None,
                 ent_target=None,
                 cand_indexes=None
                 ):
        self.input_ids = input_ids
        self.indexes = indexes
        self.bpe_len = bpe_len
        self.word_len = word_len
        self.matrix = matrix
        self.offset = offset
        self.example_id = example_id
        self.ent_target = ent_target
        self.cand_indexes = cand_indexes

def load_data(path, ent2id):
    D = {"id":[], "entities":[], "text": []}
    for data in open(path):
        d = json.loads(data)
        D["id"].append(d['doc_id'])
        D["text"].append(d['text'])
        D["entities"].append([])
        for e in d["entity"]:
            start, end, entity, label = e
            if start <= end:
                D["entities"][-1].append((start, end-1, ent2id[label]))
    return D

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, model_name='bert', max_len=512, train_stride=1, window=128, is_train=True):
        self.tokenizer = tokenizer
        self.ent2id = ent2id
        self.max_len = max_len
        self.train_stride = train_stride
        self.is_train = is_train
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
        self.mlm_probability = 0.15
        self.window = window
        self.data = self.convert(data)
        

    def __len__(self):
        return len(self.data)
    
    def get_new_ins(self, bpes, spans, indexes, cand_indexes=None, offset=None, example_id=None):
            
            bpes.append(self.sep)
            cur_word_idx = indexes[-1]
            indexes.append(0)
            
            if self.is_train:

                matrix = np.zeros((cur_word_idx, 2*self.window+1, len(self.ent2id)), dtype=np.int8)
                for _ner in spans:
                    s, e, t = _ner
                    if (e-s) <= self.window:
                        matrix[s, self.window+e-s, t] = 1
                        matrix[e, self.window-e+s, t] = 1

                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, indexes=indexes, bpe_len=len(bpes), matrix=matrix, cand_indexes=cand_indexes)
            
            else:
                ent_target = []
                for _ner in spans:
                    s, e, t = _ner
                    ent_target.append((s, e, t))
                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, indexes=indexes, bpe_len=len(bpes), word_len=cur_word_idx, offset=offset, example_id=example_id, ent_target=ent_target)

            return new_ins

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        
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

    def convert(self, path):
        
        ins_lst = []
        word2bpes = {}
        for data in open(path):
            d = json.loads(data)
            raw_words = d['text']
            example_id = d['doc_id']
            raw_ents = []

            for e in d["entity"]:
                start, end, entity, label = e
                if start <= end:
                    raw_ents.append((start, end-1, self.ent2id[label]))

            bpes = []
            indexes = []
            cand_indexes = []
            
            for idx, word in enumerate(raw_words, start=0):
                
                if word in word2bpes:
                    _bpes = word2bpes[word]
                else:
                    _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                    add_special_tokens=False)
                    word2bpes[word] = _bpes
                
                cand_indexes.append(list(range(len(bpes)+1, len(bpes)+len(_bpes)+1)))
                indexes.extend([idx]*len(_bpes))
                bpes.extend(_bpes)
            
            new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
            new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]

            for __bpes, __indexes in zip(new_bpes, new_indexes):
                spans = []
                offset = __indexes[0]
                for s, e, t in raw_ents:
                    if __indexes[0]<=s<=e<=__indexes[-1]:
                        spans += [(s-__indexes[0], e-__indexes[0], t)]

                if self.is_train:
                    __indexes = [0] + [i - offset + 1 for i in __indexes]
                    new_ins = self.get_new_ins(__bpes, spans, __indexes, cand_indexes)
                    ins_lst.append(new_ins)

                else:
                    __indexes = [0] + [i - offset + 1 for i in __indexes]
                    new_ins = self.get_new_ins(__bpes, spans, __indexes, offset=offset, example_id=example_id)
                    ins_lst.append(new_ins)

        return ins_lst
        
    def collate(self, examples):
        
        if self.is_train:

            batch_input_id, batch_index, batch_bpe_len, batch_matrix = [], [], [], []
            batch_mask_labels = []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_index.append(item.indexes)
                batch_bpe_len.append(item.bpe_len)
                batch_matrix.append(item.matrix)
                
                # WholeWordMask
                random.shuffle(item.cand_indexes)
                num_to_predict = max(1, int(round(len(item.input_ids) * self.mlm_probability)))
                masked_lms = []
                covered_indexes = set()
                for index_set in item.cand_indexes:
                    if len(masked_lms) >= num_to_predict:
                        break
                    # If adding a whole-word mask would exceed the maximum number of
                    # predictions, then just skip this candidate.
                    if len(masked_lms) + len(index_set) > num_to_predict:
                        continue
                    is_any_index_covered = False
                    for index in index_set:
                        if index in covered_indexes:
                            is_any_index_covered = True
                            break
                    if is_any_index_covered:
                        continue
                    for index in index_set:
                        covered_indexes.add(index)
                        masked_lms.append(index)
                if len(covered_indexes) != len(masked_lms):
                    raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
                mask_labels = [1 if i in covered_indexes else 0 for i in range(len(item.input_ids))]
                batch_mask_labels.append(mask_labels)

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_indexes = torch.tensor(self.sequence_padding(batch_index)).long()
            batch_bpe_lens = torch.tensor(batch_bpe_len).long()
            batch_labels = torch.tensor(self.sequence_padding(batch_matrix)).long()
            batch_masklabels = torch.tensor(self.sequence_padding(batch_mask_labels)).long()

            return self.torch_mask_tokens(batch_input_ids, batch_masklabels), batch_indexes, batch_bpe_lens, batch_labels
        
        else:
            
            batch_input_id, batch_index, batch_bpe_len, batch_word_len, batch_offset, batch_id, batch_ent_target = [], [], [], [], [], [], []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_index.append(item.indexes)
                batch_bpe_len.append(item.bpe_len)
                batch_word_len.append(item.word_len)
                batch_offset.append(item.offset)
                batch_id.append(item.example_id)
                batch_ent_target.append(item.ent_target)

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_indexes = torch.tensor(self.sequence_padding(batch_index)).long()
            batch_bpe_lens = torch.tensor(batch_bpe_len).long()
            batch_word_lens = torch.tensor(batch_word_len).long()

            return batch_input_ids, batch_indexes, batch_bpe_lens, batch_word_lens, batch_offset, batch_id, batch_ent_target

    def torch_mask_tokens(self, inputs, mask_labels=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        # labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability) if mask_labels is None else mask_labels

        special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
            ]
        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer._pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool() if mask_labels is None else probability_matrix.bool()
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs

    def __getitem__(self, index):
        item = self.data[index]
        return item
    
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)
    
    def get_evaluate_fpr(self, scores, ent_target):
        X, Y, Z = [], [], []

        for ents, score in zip(ent_target, scores):
            pred_ent = set()
            for l, start, end in zip(*np.where(score > 0)):
                pred_ent.add((l, start, end))
            ents = set(map(tuple, ents))
            X.extend(ents)
            Y.extend(pred_ent)
            Z.extend([pre_entity for pre_entity in pred_ent if pre_entity in ents])

        return X, Y, Z

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self, origins, founds, rights):
        ent2id = {"gender": 0, "education": 1, "research_interests": 2, "work_record": 3, "take_office": 4, "honorary_title": 5, "highest_education": 6, "work_for": 7, "awards": 8, "birth_place": 9, "birthday": 10, "title": 11}
        id2ent = {}
        for k, v in ent2id.items(): id2ent[v] = k
        class_info = {}
        # 
        origin_counter = Counter([id2ent[x[0]] for x in origins])
        found_counter = Counter([id2ent[x[0]] for x in founds])
        right_counter = Counter([id2ent[x[0]] for x in rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": precision, 'recall': recall, 'f1': f1, 'origin': origin, 'found': found, 'right': right}
        origin = len(origins)
        found = len(founds)
        right = len(rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1, 'origin': origin, 'found': found, 'right': right}, class_info

class RawGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, bpe_len, indexes):
        self.device = input_ids.device
        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        last_hidden_state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len) 每个关系一个矩阵
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        mask = seq_len_to_mask(lengths)  # bsz x length x length
        pad_mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs,position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        #encodr: RoBerta-Large as encoder
        #inner_dim: 64
        #ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder # bert模型
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        # self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2) #原版的dense2是(inner_dim * 2, ent_type_size * 2)
        self.dense_2 = nn.Linear(self.inner_dim * 2, self.ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, bpe_len, indexes):

        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        last_hidden_state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        outputs = self.dense_1(last_hidden_state)
        qw, kw = outputs[...,::2], outputs[..., 1::2]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(outputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        mask = seq_len_to_mask(lengths)
        logits = self.add_mask_tril(logits, mask=mask)
        return logits

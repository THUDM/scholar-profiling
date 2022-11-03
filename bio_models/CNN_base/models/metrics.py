import torch
from collections import Counter

class MetricsCalculator(object):
    def __init__(self, ent_thres, allow_nested=True):
        super().__init__()
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)
    
    def get_evaluate_fpr(self, scores, ent_target, masks):
        X, Y, Z = [], [], []
        ent_scores = scores.sigmoid()  # bsz x max_len x max_len x num_class
        ent_scores = (ent_scores + ent_scores.transpose(1, 2))/2 # 同时使用下三角的分数
        span_pred = ent_scores.max(dim=-1)[0]

        span_ents = decode(span_pred, masks, allow_nested=self.allow_nested, thres=self.ent_thres) # 解码
        for ents, span_ent, ent_pred in zip(ent_target, span_ents, ent_scores.cpu().numpy()):
            pred_ent = set()
            for s, e, l in span_ent:
                score = ent_pred[s, e]
                ent_type = score.argmax()
                if score[ent_type]>=self.ent_thres:
                    pred_ent.add((s, e, ent_type))
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
        origin_counter = Counter([id2ent[x[-1]] for x in origins])
        found_counter = Counter([id2ent[x[-1]] for x in founds])
        right_counter = Counter([id2ent[x[-1]] for x in rights])
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
    
def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area.
    """
    for start in range(seq_len):
        for end in range(start, seq_len):
            yield (start, end)

# 按token解码
def decode(scores, masks, allow_nested=False, thres=0.5):
    batch_chunks = []
    for idx, (curr_scores, mask) in enumerate(zip(scores, masks)):
        curr_len = int(sum(mask))
        curr_non_mask = scores.new_ones(curr_len, curr_len, dtype=bool).triu()
        tmp_scores = curr_scores[:curr_len, :curr_len][curr_non_mask].cpu().numpy()  # -1 x 2
        confidences, label_ids = tmp_scores, tmp_scores>=thres
        labels = [i for i in label_ids]
        chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != 0]
        confidences = [conf for label, conf in zip(labels, confidences) if label != 0]

        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_nested=allow_nested)

        if len(chunks):
            batch_chunks.append(set([(s, e, l) for l,s,e in chunks]))
        else:
            batch_chunks.append(set())
    return batch_chunks


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks, allow_nested: bool=True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)

    return filtered_chunks

import torch
from collections import Counter
import collections

class MetricsCalculator(object):
    def __init__(self, ent_thres, id2ent,allow_nested=True):
        super().__init__()
        self.allow_nested = allow_nested
        self.ent_thres = ent_thres
        self.id2ent = id2ent

    def roll_tensor_columns_efficient(self, original_tensor):
        tensor = original_tensor.clone()
        bs, N, C = tensor.shape[:3]
        W = C // 2

        indices = torch.arange(N, device=original_tensor.device).unsqueeze(0).unsqueeze(2)
        shifts = torch.arange(-W, W + 1, device=original_tensor.device).unsqueeze(0).unsqueeze(1)
        rolled_indices = (indices - shifts + N) % N

        if len(tensor.shape) == 4:
            rolled_indices = rolled_indices.unsqueeze(-1)
        
        rolled_tensor = torch.gather(tensor, 1, rolled_indices.expand(tensor.size()))

        return rolled_tensor

    def get_evaluate_fpr_overlap(self, examples, scores, word_len, offset, example_id):
        
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(example_id):
            features_per_example[example_id_to_index[feature]].append(i)

        index_to_example_id = {}
        for k, v in example_id_to_index.items(): index_to_example_id[v] = k

        X, Y, Z = [], [], []
        for example_index, example in enumerate(examples["entities"]):
            example_annotations = set(map(tuple, example))
            example_predictions = set()

            feature_indices = features_per_example[example_index]
            for feature_index in feature_indices:
                
                predictions = set()
                remove = set()

                ent_scores = scores[feature_index].unsqueeze(dim=0).sigmoid()
                ent_scores = (ent_scores + torch.flip(self.roll_tensor_columns_efficient(ent_scores), dims=[2]))/2
                span_pred = ent_scores.max(dim=-1)[0]

                span_ents = decode(span_pred, [word_len[feature_index]], allow_nested=self.allow_nested, thres=self.ent_thres)
                ent_preds = ent_scores.argmax(dim=-1)

                for span_ent, ent_pred in zip(span_ents, ent_preds):
                    for s, e, s_o, e_o in span_ent:
                        ent_type = ent_pred[s_o, e_o]
                        flag = False
                        for ex_ck in example_predictions:
                            s1, e1, t = ex_ck
                            if (s1 <= e+offset[feature_index] and s+offset[feature_index] <= e1 and t==ent_type.item()):
                                flag = True
                                remove.add(ex_ck)
                                predictions.add((min(s1, s+offset[feature_index]), max(e1, e+offset[feature_index]), ent_type.item()))
                        if not flag:
                            predictions.add((s+offset[feature_index], e+offset[feature_index], ent_type.item()))
                            
                for en in remove:
                    example_predictions.remove(en)
                example_predictions.update(predictions)

            X.extend(example_annotations)
            Y.extend(example_predictions)
            Z.extend([pre_entity for pre_entity in example_predictions if pre_entity in example_annotations])
            
        return X, Y, Z

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self, origins, founds, rights):
        class_info = {}
        origin_counter = Counter([self.id2ent[x[-1]] for x in origins])
        found_counter = Counter([self.id2ent[x[-1]] for x in founds])
        right_counter = Counter([self.id2ent[x[-1]] for x in rights])
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

def decode(scores, length, allow_nested=False, thres=0.5):
    batch_chunks = []
    for idx, (curr_scores, curr_len) in enumerate(zip(scores, length)):
        w = curr_scores.size(1)//2
        tmp_scores = curr_scores[:curr_len, w:]
        tmp = (tmp_scores>=thres)
        chunks = tmp.nonzero(as_tuple=True)
        confidences = curr_scores[chunks].tolist()
        chunks = tmp.nonzero()
        assert len(confidences) == len(chunks)
        chunks = [ck for _, ck in sorted(zip(confidences, chunks.tolist()), reverse=True)]
        chunks, origin_chunks = filter_clashed_by_priority(chunks, curr_len, allow_nested=allow_nested)
        if len(chunks):
            batch_chunks.append(set([(s, e, s_o, e_o+w) for (s, e), (s_o, e_o) in zip(chunks, origin_chunks)]))
        else:
            batch_chunks.append(set())
    return batch_chunks

def is_overlapped(chunk1: tuple, chunk2: tuple):
    (s1, e1), (s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (s1, e1), (s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)

def filter_clashed_by_priority(chunks, len, allow_nested: bool=True):
    filtered_chunks = []
    origin_chunks = []
    for ck in chunks:
        s, e = ck
        if s+e<len and (s,s+e) not in filtered_chunks:
            if all(not is_clashed((s,s+e), ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
                filtered_chunks.append((s,s+e))
                origin_chunks.append(ck)

    return filtered_chunks, origin_chunks

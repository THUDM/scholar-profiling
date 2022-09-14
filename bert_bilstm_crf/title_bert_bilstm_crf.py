import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

title_list2Id={ "Other(其他)": 0,
    "B-Professor(教授)": 1,
    "I-Professor(教授)": 2,
    "B-Researcher(研究员)": 3,
    "I-Researcher(研究员)": 4,
    "B-Associate Professor(副教授)": 5,
    "I-Associate Professor(副教授)": 6,
    "B-Assistant Professor(助理教授)": 7,
    "I-Assistant Professor(助理教授)": 8,
    "B-Professorate Senior Engineer(教授级高级工程师)": 9,
    "I-Professorate Senior Engineer(教授级高级工程师)": 10,
    "B-Engineer(工程师)": 11,
    "I-Engineer(工程师)": 12,
    "B-Lecturer(讲师)": 13,
    "I-Lecturer(讲师)": 14,
    "B-Senior Engineer(高级工程师)": 15,
    "I-Senior Engineer(高级工程师)": 16,
    "B-Ph.D(博士生)": 17,
    "I-Ph.D(博士生)": 18,
    "B-Associate Researcher(副研究员)": 19,
    "I-Associate Researcher(副研究员)": 20,
    "B-Assistant Researcher(助理研究员)": 21,
    "I-Assistant Researcher(助理研究员)": 22,
    "B-Student(学生)": 23,
    "I-Student(学生)": 24,
    "<START>": 25,
    "<STOP>": 26
    }

#####################################################################
# Helper functions to make the code more readable.

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    if len(content) == 1: # ‘O’
        return tag_class
    ht = content[-1]
    return tag_class, ht

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default1 = tags["Other(其他)"]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # sep (s,)
        # End of a chunk 1 实体的后边界
        if tok == default1 and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1:
            res = get_chunk_type(tok, idx_to_tag) # 获取对应tok的类型
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast))) 


START_TAG = "<START>"
STOP_TAG = "<STOP>"

# 可以提高2个百分点 最大的问题是提取的文本的标签与个人的对应
class Bert_LSTM_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.tag_to_ix = title_list2Id
        self.tagset_size = len(title_list2Id)
        self.emb_dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(config.hidden_size, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[title_list2Id[START_TAG], :] = -10000
        self.transitions.data[:, title_list2Id[STOP_TAG]] = -10000


    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.)#.to('cuda')
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1).to("cuda")
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha


    def _get_features(self, input_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            label_tags: (bs, seq_len)
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ) 
        bert_embs = outputs[0]

        word_reps = self.emb_dropout(bert_embs)
        packed_embs = pack_padded_sequence(word_reps, attention_mask.sum(-1).cpu(), batch_first=True,
                                           enforce_sorted=False)
        packed_outs, (hidden, _) = self.lstm(packed_embs)
        sequence_output, _ = pad_packed_sequence(packed_outs, batch_first=True,
                                                 total_length=attention_mask.sum(-1).max())

        lstm_feats = self.hidden2tag(sequence_output)

        return lstm_feats


    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        #feats = feats.transpose(0,1)

        score = torch.zeros(tags.shape[0]).to('cuda')
        tags = torch.cat([torch.full([tags.shape[0],1],self.tag_to_ix[START_TAG], dtype=torch.long).to("cuda"),tags],dim=1)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:,-1]]
        return score

    def _viterbi_decode_new(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)#.to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1]).to("cuda")
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood_parallel(self, token_ids, mask, tags):
        feats = self._get_features(token_ids, mask)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags)
        return torch.sum(forward_score - gold_score) / token_ids.size(0)

    def forward(self, token_ids, masks):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        scores, pre_tag_seq, pred_titles =[], [], []
        feats = self._get_features(token_ids, masks)
        for feat, mask in zip(feats, masks):
            score, tag_seq = self._viterbi_decode_new(feat)
            tag_seq = (torch.tensor(tag_seq).to("cuda") * mask).tolist()
            pred_title = get_chunks(tag_seq, title_list2Id)
            scores.append(score)
            pre_tag_seq.append(tag_seq)
            pred_titles.append(pred_title)
        # # Find the best path, given the features.
        # score, tag_seq = self._viterbi_decode_new(feats)
        return scores, pre_tag_seq, pred_titles

from torch import nn
from fastNLP import seq_len_to_mask
from torch_scatter import scatter_max
import torch
import torch.nn.functional as F
from .cnn import CrossTransformer

class CNNNer(nn.Module):
    def __init__(self, encoder, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, chunks_size=128, cnn_depth=3, attn_dropout=0.15, use_tri_bias=True):
        super(CNNNer, self).__init__()
        self.pretrain_model = encoder
        hidden_size = self.pretrain_model.config.hidden_size

        if size_embed_dim!=0:
            n_pos = 50
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        
        self.cnn_dim = cnn_dim
        biaffine_input_size = hidden_size

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.GELU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(0.4)
        self.chunks_size = chunks_size

        self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size + 1, biaffine_size + 1))
        torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)

        layers = []
        if cnn_depth > 0:
            for i in range(cnn_depth):
                layers.append(CrossTransformer(dim=cnn_dim, dropout=attn_dropout, use_tri_bias=use_tri_bias, scale=False))
            self.cross_layers = nn.Sequential(*layers)

        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop
        self.num_ner_tag = num_ner_tag

    def _chunk(self, x, w):
        '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)
    
    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded
    
    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)


    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _get_invalid_locations_mask(self, w: int, d: int):
        affected_seq_len = w * d
        mask = self._get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, :]

        ending_mask = mask.flip(dims=(1, 2)).bool().cuda()
        return affected_seq_len, mask.bool().cuda(), ending_mask

    def forward(self, input_ids, bpe_len, indexes, matrix=None): 
        attention_mask = seq_len_to_mask(bpe_len)
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']

        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]
        lengths, _ = indexes.max(dim=-1)
        mask_seq = seq_len_to_mask(lengths)

        _, n, _ = state.size()
        padding_len = 0
        
        chunks_size_ = self.chunks_size * 2
        if n % chunks_size_ != 0:
            padding_len = (chunks_size_ - n % chunks_size_) % chunks_size_
            state = F.pad(state, (0,0,0,padding_len), value=0)
            mask_seq = F.pad(mask_seq, (0,padding_len), value=0)
            n += padding_len
        chunks_count = n // self.chunks_size
        chunk_state = self._chunk(state, self.chunks_size)
        
        head_state = self.head_mlp(chunk_state)
        tail_state = self.tail_mlp(chunk_state)
        
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        scores1 = torch.einsum('bcxi, oij, bcyj -> bocxy', head_state, self.U, tail_state)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(3).expand(-1, -1, -1, tail_state.size(2), -1),
                                 self.dropout(tail_state).unsqueeze(2).expand(-1, -1, head_state.size(2), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat,
                                     self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bcmnh,kh->bkcmn', affined_cat, self.W)
        scores = scores2 + scores1

        bs, dim, chunks, l, _ = scores.size()
        w = l//2
        chunk_scores = scores.reshape(bs*dim, chunks, l, l)

        # convert diagonals into columns
        diagonal_chunk_scores = self._pad_and_transpose_last_two_dims(chunk_scores, padding=(0, 0, 0, 1))

        diagonal_scores = diagonal_chunk_scores.new_empty((bs * dim, chunks_count, w, w * 2 + 1)) # 2
        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_scores[:, :-1, :, w:] = diagonal_chunk_scores[:, :, :w, :w + 1]
        diagonal_scores[:, -1, :, w:] = diagonal_chunk_scores[:, -1, w:, :w + 1]
        # - copying the lower triangle
        diagonal_scores[:, 1:, :, :w] = diagonal_chunk_scores[:, :, - (w + 1):-1, w + 1:]
        diagonal_scores[:, 0, 1:w, 1:w] = diagonal_chunk_scores[:, 0, :w - 1, 1 - w:]

        # separate bsz and num_heads dimensions again
        diagonal_scores = diagonal_scores.view(bs, dim, n, 2 * w + 1).permute(0, 2, 3, 1)

        remove_from_windowed_attention_mask = (mask_seq != 1)[:, :, None, None]
            
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            remove_from_windowed_attention_mask.new_ones(size=remove_from_windowed_attention_mask.size()), remove_from_windowed_attention_mask, w
        )
        diagonal_mask.masked_fill_(remove_from_windowed_attention_mask, 1.0)
        pad_mask = diagonal_mask.transpose(2, 3)
        
        
        if padding_len > 0:
            diagonal_scores = diagonal_scores[:, :-padding_len, :, :]
            pad_mask = pad_mask[:, :-padding_len, :, :]

        if hasattr(self, 'cross_layers'):
            u_scores = diagonal_scores.masked_fill_(pad_mask, 0)
            u_scores = self.cross_layers((u_scores, pad_mask.squeeze(-1).contiguous()))[0]
            scores = u_scores + diagonal_scores
            
        scores = self.down_fc(scores)
        scores.masked_fill_(scores.isinf() | scores.isnan(), 0)

        if self.training:
            assert scores.size(-1) == matrix.size(-1)
            pad_mask = pad_mask.expand(scores.size())
            flat_scores = scores.reshape(-1)
            flat_matrix = matrix.reshape(-1)
            mask = pad_mask.reshape(-1).eq(0).float().view(input_ids.size(0), -1)
            flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
            
            loss = ((flat_loss.view(input_ids.size(0), -1)*mask).sum(dim=-1)).mean()
        
            return {'loss': loss}
        
        scores.masked_fill_(pad_mask, float(-1e-6))

        return {'scores': scores}
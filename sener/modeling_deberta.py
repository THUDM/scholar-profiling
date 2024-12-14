import torch
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, build_relative_position, DebertaV2Model, DebertaV2Encoder
import torch.nn.functional as F

class Arrow_Attention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config, one_sided_attn_window_size):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

        self.one_sided_attn_window_size = one_sided_attn_window_size

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    @staticmethod
    def _chunk(x, w):
        '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

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

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

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

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.BoolTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        w_size_ = self.one_sided_attn_window_size * 2
        padding = 0 
        if hidden_states.size(1) % w_size_ != 0:
            padding = ((w_size_ - hidden_states.size(1) % w_size_) % w_size_)
            hidden_states = F.pad(hidden_states, (0,0,0,padding), value=0)
            attention_mask = F.pad(attention_mask, (0,padding), value=0)

        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        global_idx = [0]
        global_len = len(global_idx)
        idx_mask = torch.zeros(hidden_states.size(1), dtype=torch.bool).to(hidden_states.device)
        idx_mask[global_idx] = True

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)

        # normalize query
        query_vectors = query_layer.view(-1, self.num_attention_heads, query_layer.size(-2), query_layer.size(-1)).transpose(1, 2) / scale.to(dtype=query_layer.dtype)
        key_vectors = key_layer.view(-1, self.num_attention_heads, key_layer.size(-2), key_layer.size(-1)).transpose(1, 2)
        value_vectors = value_layer.view(-1, self.num_attention_heads, key_layer.size(-2), key_layer.size(-1)).transpose(1, 2)

        attention_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )
        
        # values to pad for attention probs
        is_index_masked = (attention_mask == 0)
        global_mask = attention_mask[:, idx_mask]
        remove_from_windowed_attention_mask = (attention_mask != 1)[:, :, None, None]
        remove_from_windowed_attention_mask[:, idx_mask] = True

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attention_scores += diagonal_mask
        attention_scores = attention_scores.transpose(1, 2).view(-1, attention_scores.size(1), attention_scores.size(-1))

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att, cls_att, global_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, self.one_sided_attn_window_size, idx_mask, global_len
            )

        cls_attention_scores = torch.einsum("bxd,byd->bxy", (query_layer, key_layer[:,idx_mask] / scale.to(dtype=query_layer.dtype)))
        global_attention_scores = torch.einsum("bxd,byd->bxy", (query_layer[:,idx_mask], key_layer / scale.to(dtype=query_layer.dtype)))

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
            cls_attention_scores = cls_attention_scores + cls_att
            global_attention_scores = (global_attention_scores + global_att) * (torch.log(torch.tensor(query_layer.size(1), dtype=torch.float)) / torch.log(torch.tensor(512.)))

        cls_attention_scores = cls_attention_scores.view(-1, self.num_attention_heads, cls_attention_scores.size(-2), cls_attention_scores.size(-1))
        cls_attention_scores.masked_fill_(~global_mask[:, None, None, :], torch.finfo(cls_attention_scores.dtype).min)
        cls_attention_scores = cls_attention_scores.view(-1, cls_attention_scores.size(-2), cls_attention_scores.size(-1))

        attention_scores = torch.cat((cls_attention_scores, attention_scores), dim=-1)

        # free memory
        del cls_attention_scores
        
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        ).transpose(1, 2)

        attention_probs = nn.functional.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attention_probs = torch.masked_fill(attention_probs, is_index_masked[:, :, None, None], 0.0)
        attention_probs = attention_probs.type_as(attention_scores)

        # free memory
        del attention_scores
        
        attention_probs = self.dropout(attention_probs)

        context_layer = self._sliding_chunks_matmul_attn_probs_value(attention_probs[:,:,:,global_len:], value_vectors, self.one_sided_attn_window_size)

        cls_context = torch.einsum("bwcd,bdch->bwch", (attention_probs[:,:,:,:global_len], value_vectors[:,idx_mask]))
        context_layer = context_layer + cls_context
        
        global_attention_scores = global_attention_scores.view(
            -1, self.num_attention_heads, global_attention_scores.size(-2), global_attention_scores.size(-1)
        ).transpose(1, 2)
        global_attention_scores.masked_fill_(~global_mask[:, :, None, None], torch.finfo(global_attention_scores.dtype).min)
        global_attention_scores = global_attention_scores.masked_fill(
            is_index_masked[:, None, None, :],
            torch.finfo(global_attention_scores.dtype).min,
        )

        # compute global attn probs
        global_attention_scores = nn.functional.softmax(
            global_attention_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability
        
        global_attention_scores = self.dropout(global_attention_scores)
        global_context = torch.einsum("bwcd,bdch->bwch", (global_attention_scores, value_vectors))

        context_layer[:,idx_mask] = global_context
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.contiguous().view(new_context_layer_shape)

        if padding > 0:
            context_layer = context_layer[:, :-padding, :]

        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, chunk_size, global_mask, global_len):
        
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(
                q,
                key_layer.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )

        global_relative_pos = relative_pos[:,global_mask]
        cls_relative_pos = relative_pos[:,:,global_mask]
        chunk_relative_pos = torch.cat((relative_pos[:,-1,-chunk_size-1:-1],relative_pos[:,0,:chunk_size+1]), dim=-1).unsqueeze(0)
        # convert diagonals into columns

        att_span = self.pos_ebd_size
        chunk_relative_pos = chunk_relative_pos.long().to(query_layer.device)
        cls_relative_pos = cls_relative_pos.long().to(query_layer.device)
        global_relative_pos = global_relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
                query_layer.size(0) // self.num_attention_heads, 1, 1
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)

        score = 0
        cls_score = 0
        global_score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(chunk_relative_pos + att_span, 0, att_span * 2 - 1)
            cls_c2p_pos = torch.clamp(cls_relative_pos + att_span, 0, att_span * 2 - 1)
            global_c2p_pos = torch.clamp(global_relative_pos + att_span, 0, att_span * 2 - 1)
            chunk_c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), chunk_relative_pos.size(-1)]),
            )
            cls_c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=cls_c2p_pos.expand([query_layer.size(0), query_layer.size(1), global_len]),
            )
            global_c2p_att = torch.gather(
                c2p_att,
                dim=-1,
                index=global_c2p_pos.expand([query_layer.size(0), global_len, query_layer.size(1)]),
            )
            score += chunk_c2p_att / scale.to(dtype=c2p_att.dtype)
            cls_score += cls_c2p_att / scale.to(dtype=c2p_att.dtype)
            global_score += global_c2p_att / scale.to(dtype=c2p_att.dtype)

        # position->content
        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
            p2c_pos = torch.clamp(-chunk_relative_pos + att_span, 0, att_span * 2 - 1)
            cls_p2c_pos = torch.clamp(-cls_relative_pos + att_span, 0, att_span * 2 - 1)
            global_p2c_pos = torch.clamp(-global_relative_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            chunk_p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=p2c_pos.expand([query_layer.size(0), key_layer.size(-2), p2c_pos.size(-1)]),
            )
            cls_p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=cls_p2c_pos.expand([query_layer.size(0), key_layer.size(-2), global_len]),
            ).transpose(-1, -2)
            global_p2c_att = torch.gather(
                p2c_att,
                dim=-1,
                index=global_p2c_pos.expand([query_layer.size(0), global_len, key_layer.size(-2)]),
            ).transpose(-1, -2)
            chunk_p2c_att = self.roll_tensor_columns_efficient(chunk_p2c_att).flip(dims=[2])
            score += chunk_p2c_att / scale.to(dtype=p2c_att.dtype)
            global_score += cls_p2c_att / scale.to(dtype=p2c_att.dtype)
            cls_score += global_p2c_att / scale.to(dtype=p2c_att.dtype)

        return score, cls_score, global_score

class DebertaModel(DebertaV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = DebertaEncoder(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = Arrow_Attention(config, 128)
            
class DebertaEncoder(DebertaV2Encoder):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__(config)

    def get_attention_mask(self, attention_mask):
        pass
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        pass
        return relative_pos

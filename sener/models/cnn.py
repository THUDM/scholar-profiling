import math
from torch import nn
import torch
import numpy as np


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x

class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                bias=False, groups=groups)

    def forward(self, x, mask):
        
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x
    
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int, padding_idx=None,
            scale=False
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)
        if scale:
            self.weight.data = self.weight.data / math.sqrt(embedding_dim)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)

class CrossTransformer(nn.Module):
    def __init__(self, dim, dropout=0.3, use_tri_bias=True, scale=False):
        super().__init__()
        self.h_dim = dim
        self.use_tri_bias = use_tri_bias
        if use_tri_bias is True:
            self.pos_embed = nn.Embedding(2, dim)
            nn.init.xavier_normal_(self.pos_embed.weight.data, gain=0.1 if scale else 1)
        if use_tri_bias is 2:
            pos = torch.ones(512, 512, dtype=torch.long) * 2
            pos.triu_()
            pos = pos - torch.eye(512)
            self.register_buffer('pos', pos.long())
            self.pos_embed = nn.Embedding(3, dim)
            nn.init.xavier_normal_(self.pos_embed.weight.data, gain=0.1 if scale else 1)

        self.h_qkv = nn.Linear(dim, 3 * dim)
        self.v_qkv = nn.Linear(dim, 3 * dim)
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            512,
            dim,
            scale=scale
        )
        self.embed_positions.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

        self.dense = nn.Linear(2 * dim, dim, bias=True)
        self.LayerNorm = nn.LayerNorm(dim)

        self.conv1 = MaskConv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = MaskConv2d(dim, dim, kernel_size=3, padding=1)
        self.LayerNorm2 = nn.LayerNorm(dim)

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

    def forward(self, x):
        """

        :param x: (bsz x max_len x max_len x dim, bsz x max_len x max_len)
        :return:
        """
        x, mask = x
        x = x.clamp(-1000, 1000)
        bsz, max_len, w, dim = x.size()
        sinusoidal_pos = self.embed_positions(w, 0)[
                         None, :, :
                         ].chunk(2, dim=-1)

        if self.use_tri_bias:
            pos_mask = torch.ones(max_len, w, dtype=torch.long, device=x.device)
            pos_mask[:, :w//2] = 0
            pos = self.pos_embed(pos_mask)[None]
            x = pos + x

        # horizontal
        h_mask = mask.view(-1, w).sum(dim=-1) != w
        h_tmp_mask = (mask.view(-1, 1, w)[h_mask]).bool()
        h_attn_mask = mask.view(-1, 1, w)[h_mask].float() * -10000.0
        h_scores = x.reshape(-1, w, dim)
        _h_scores = h_scores[h_mask]
        __h_scores = self.h_qkv(_h_scores).clamp(-10000, 10000)
        h_q, h_k, h_v = __h_scores.chunk(3, dim=-1)
        h_q, h_k = self.apply_rotary(h_q, sinusoidal_pos), self.apply_rotary(h_k, sinusoidal_pos)
        h_attn = torch.matmul(h_q, h_k.transpose(-1, -2)) / self.scale
        h_attn = h_attn.clamp(-10000, 10000) + h_attn_mask
        h_attn = self.dropout(torch.softmax(h_attn, dim=-1)).masked_fill(h_tmp_mask, 0)
        h_v = torch.matmul(h_attn, h_v)

        # vertical
        t_mask = mask
        v_mask = t_mask.reshape(-1, w).sum(dim=-1) != w
        v_tmp_mask = (t_mask.view(-1, 1, w)[v_mask]).bool()
        v_attn_mask = t_mask.reshape(-1, 1, w)[v_mask].float() * -10000.0
        v_scores = torch.flip(self.roll_tensor_columns_efficient(x), dims=[2]).reshape(-1, w, dim)
        _v_scores = v_scores[v_mask]
        _v_scores = self.v_qkv(_v_scores).clamp(-10000, 10000)
        v_q, v_k, v_v = _v_scores.chunk(3, dim=-1)
        v_q, v_k = self.apply_rotary(v_q, sinusoidal_pos), self.apply_rotary(v_k, sinusoidal_pos)
        v_attn = torch.matmul(v_q, v_k.transpose(-1, -2)) / self.scale
        v_attn = v_attn.clamp(-10000, 10000) + v_attn_mask
        v_attn = self.dropout(torch.softmax(v_attn, dim=-1)).masked_fill(v_tmp_mask, 0)
        v_v = torch.matmul(v_attn, v_v)

        v = torch.cat([h_v, v_v], dim=-1)
        v = self.dense(v)
        v = self.dropout(v)
        _x = torch.zeros_like(x).reshape(-1, w, dim)
        _x[h_mask] = v.to(_x)
        v = self.LayerNorm(_x + x.reshape(-1, w, dim))
        v = v.reshape(bsz, max_len, w, dim).permute(0, 3, 1, 2)

        c_v = self.conv1(v, mask[:, None])
        c_v = self.act(c_v)
        c_v = self.conv2(c_v, mask[:, None]).permute(0, 2, 3, 1).reshape(-1, w, dim)
        c_v = self.dropout(c_v) + v.permute(0, 2, 3, 1).reshape(-1, w, dim)

        v = self.LayerNorm2(c_v)

        return (v.reshape(bsz, max_len, w, dim), mask)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)

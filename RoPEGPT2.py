from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import math

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim_head):
        super().__init__()
        self.dim_head = dim_head

    def forward(self, pos):
        # 生成位置编码
        position_indices = torch.arange(0, pos, dtype=torch.float).unsqueeze(1)
        indices = torch.arange(0, self.dim_head // 2, dtype=torch.float).unsqueeze(0)
        
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / self.dim_head)
        angle_rads = position_indices * angle_rates
        sines = torch.sin(angle_rads)
        cosines = torch.cos(angle_rads)
        
        pos_encoding = torch.stack((sines, cosines), dim=2).flatten(1)
        return pos_encoding

class RoPEGPT2Attention(GPT2Attention):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.rope = RoPEPositionalEncoding(config.n_embd)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        seq_length = query.size(-2)
        pos_encoding = self.rope(seq_length).to(query.device)
        
        # 应用RoPE
        query, key = self.apply_rope(query, key, pos_encoding)
        
        return super()._attn(query, key, value, attention_mask, head_mask)

    def apply_rope(self, query, key, pos_encoding):
        # 确保pos_encoding的形状正确
        # 增加维度以匹配query和key的维度，特别是批处理维度和头部维度
        cos = pos_encoding[:, :query.size(-1) // 2].cos()[None, None, :, :]
        sin = pos_encoding[:, :query.size(-1) // 2].sin()[None, None, :, :]

        # 分别对query和key应用RoPE编码
        q_cos = query[..., ::2] * cos + query[..., 1::2] * sin
        q_sin = query[..., 1::2] * cos - query[..., ::2] * sin
        query_rot = torch.stack((q_cos, q_sin), dim=-1).reshape_as(query)

        k_cos = key[..., ::2] * cos + key[..., 1::2] * sin
        k_sin = key[..., 1::2] * cos - key[..., ::2] * sin
        key_rot = torch.stack((k_cos, k_sin), dim=-1).reshape_as(key)

        return query_rot, key_rot

# 替换原有的attention

class RoPEGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # 替换原有的attention模块
        for i, block in enumerate(self.transformer.h):
            self.transformer.h[i].attn = RoPEGPT2Attention(config)

config = GPT2Config()
model = RoPEGPT2Model(config)

print(model)
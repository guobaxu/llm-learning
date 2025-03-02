import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len].detach()
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        # hid_dim: 每个词输出的向量维度
        # n_heads: 注意力头数量
        # dropout: 丢弃率
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        # 每个注意力头的维度
        self.dim_heads = hid_dim // n_heads
        # 定义缩放系数
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
        # torch.sqrt输入是tensor，所以需要将hid_dim // n_heads转化为tensor
        # self.scale = torch.sqrt(self.dim_heads)
        # 定义W_q, W_k, W_v矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        # 定义全连接层
        self.fc = nn.Linear(hid_dim, hid_dim)
        # dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: [Batchsize, length, channel]
        # mask:
        B = query.shape[0]
        # 计算Q, K, V矩阵
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 把Q, K, V拆分
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        # Q, K, V: [B, n_heads, L, dim_heads]
        Q = Q.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        K = K.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        V = V.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        
        # 第1步：Q 乘以 K的转置，除以scale
        # attention: [B, n_heads, L_Q, L_K]
        attention = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        # 如果 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10，
        # 这里用“0”来指示哪些位置的词向量不能被attention到，比如padding位置，
        # 当然也可以用“1”或者其他数字来指示，主要设计下面2行代码的改动。
        if mask is not None:
            attention = attention.masked_fill(mask==0, -1e10)
            # attention.masked_fill(mask==0, -1e10)?
        
        # 第2步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，即对得分矩阵的行向量（即每一个query向量对key向量得到score）进行归一化
        attention = self.dropout(torch.softmax(attention, dim=-1))

        # 第3步，attention结果与V相乘，得到多头注意力的结果
        # x:[B, n_heads, L_Q, dim_heads]
        x = torch.matmul(attention, V)

        # 第4步：将x转置拼接在一起，并进行最后的全连接层转换
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, self.n_heads * self.dim_heads)
        # 最后的全链接层
        x = self.fc(x)

        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        # 注意力头维度
        self.dim_heads = hid_dim // n_heads
        # 缩放系数
        self.scale = torch.sqrt(torch.FloatTensor(self.dim_heads))

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.w_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)   # torch.dropout VS nn.Dropout?
    
    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        # Q:[B, L, hid_dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 拆分，转置
        # Q:[B, n_heads, L, dim_head]
        Q = Q.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        K = K.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        V = V.view(B, -1, self.n_heads, self.dim_heads).permute(0,2,1,3)
        # 计算att
        # attention:[B, n_heads, L, L]
        attention = torch.matmul(Q, K.permute(0,1,2))/self.scale
        # mask
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        # softmax
        attention = self.dropout(torch.softmax(attention, dim=-1))
        # 加权
        # x:[B, n_heads, L, dim_head]
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, -1, self.n_heads * self.dim_heads)
        # 最后的全连接层输出
        x = self.w_o(x)
        return x
# batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
query = torch.rand(64, 12, 300)
# batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
key = torch.rand(64, 10, 300)
# batch_size 为 64，有 10 个词，每个词的 Value 向量是 300 维
value = torch.rand(64, 10, 300)
attention = MultiHeadAttention(hid_dim=300, n_heads=6, dropout=0.1)
output = attention(query, key, value)
## output: torch.Size([64, 12, 300])
print(output.shape)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    """
    基础的Encoder-Decoder结构。
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "定义生成器，由linear和softmax组成"
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

"""
Encoder部分
"""
def clones(module, N):
    "产生N个完全相同的网络层"
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "完整的Encoder包含N层"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "每一层的输入是x和mask"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)








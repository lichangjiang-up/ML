import copy
import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0
        )
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)

    def forward(self, x):
        """该函数为向前传播逻辑，所有层都有。
        当传给该类对象实例化参数时，自动调用该函数。
        参数x: embedding为首层, x为输入token下标张量"""
        return self.embed(x) * self.sqrt_embedding_dim


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, p_dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)

        # 初始化一个位置编码0矩阵: size(max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        # 初始化一个绝对索引矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变换矩阵，实现跳跃变化
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-1.0) / embedding_dim)

        # 将前面定义的变换矩阵及奇数偶数分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将二维张量转换为三维张量
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册成模型buffer，这个buffer不是模型中的参数，不跟随优化器同步更新
        # 注册成buffer后我们就可以在模型保存后重新加载时，将这个位置编码器和模型参数一同加载进来。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: 代表文本序列的词嵌入表示
        # 首先明确pe编码太长了，将第二个维度，也就是max_len对应的那个维度缩小成x的句子长度。
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    """生成向后的掩码张量，
    参数size是掩码张量的最后的两个维度，
    它的最后两维形成一个方阵。"""

    return np.tril(np.ones((1, size, size)), 0).astype("uint8")


# Attention(Q, K, V) = Softmax(Q*KT/sqrt_dk)*V
def attention(Q, K, V, mask=None, dropout=None):
    """注意力机制的实现，输入分别是query、key、value、mask：掩码张量，
    dropout 是nn.Dropout层的实例化对象，默认为None"""

    # 在函数中，首先取query最后一维的大小，一般情况下就等同于我们的词嵌入维度命名为d_k
    d_k = Q.size(-1)

    # 按照注意力公式，将Q与K的转置相乘，这里的K是将最后两个维度进行转置再除以缩放系数
    # 得到注意力得分张量scores
    scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(d_k))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作
    p_attn = scores.softmax(dim=-1)

    # 之后判断是否使用dropout进行随机设置0
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后，根据公式将p_attn与value张量相乘获得最终的query
    return torch.matmul(p_attn, V), p_attn


# 多头注意力机制下，要用到多个结构相同的线性层
# 需要使用clone函数把他们初始化到一个网络层列表的对象中
def clones(module, N):
    # module: 代表要克隆的目标网络层
    # N: 复制份数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MutiheaderAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head: 代表几个头的参数
        # embedding 代表词嵌入维度
        super(MutiheaderAttention, self).__init__()

        # 要确认一个事实：多头数量header需要整除词嵌入的维度embedding_dim
        assert embedding_dim % head == 0

        # 得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head

        self.head = head
        self.embeding_dim = embedding_dim

        # 获得线性层，需要获得4个，分别是Q, K, V 以及最终输出线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # 初始化注意力张量
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key  value 是注意力机制的三个输入张量, mask 代表掩码张量
        # 首先判断是否使用掩码张量
        if mask is not None:
            # 使用squeeze将掩码张量进行维度扩充，代表多头重的第n个头
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        def view_trans(model, x):
            return model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        # 首先事宜zip将网络层和输入数据连接在一起，模型的输出可以用view和transpose进行维度和形状的转换
        query, key, value = [
            view_trans(model, x) for model, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            Q=query, K=key, V=value, mask=mask, dropout=self.dropout
        )
        # 每个头的计算张量是四维张量，需要转换形状
        # 前面一经将1,2两个维度进行过转置，在这里需要转换回来
        x = x.transpose(1 ,2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

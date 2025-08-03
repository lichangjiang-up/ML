import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
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
    def __init__(self, embedding_dim: int, p_dropout=0.1, max_len=512):
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
    def __init__(self, head: int, embedding_dim: int, dropout=0.1):
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
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)


# 需要添加两个全连接层
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)


# nn.LayerNorm 规范化层：:稳定自注意力层的输出，防止梯度爆炸

# class LayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         """初始化函数有两个参数，一个是features，表示词嵌入的维度，
#         另一个是eps它是一个足够小的数，在规范化公式的分母中出现，
#         防止分母为0."""
#         super(LayerNorm, self).__init__()

#         # 根据features的形状初始化两个参数张量a2, 和b2, 一个全是0一个全是1
#         self.a2 = nn.Parameter(torch.ones(features))
#         self.b2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps

#     def forward(self, x):
#         """输入参数x代表来自上一层的输出"""
#         # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度和输入维度一致
#         # 接着再求一个维度的标准差，然后就是根据规范化公式，
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size: 代表词嵌入的维度
        # dropout：进行Dropout操作的置零比例
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        nx = self.norm(x)
        nx = sublayer(nx)
        nx = self.dropout(nx)
        return x + nx


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn, feed_forward, dropout=0.1):
        """
        size: 词嵌入维度
        self_attn: 自注意层实例
        feed_forward: 前馈前连接层实例
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size: int, self_attn, src_attn, feed_farward, dropout):
        """
        size: 词嵌入维度大小
        self_attn: 多头自注意力对象Q=K=V
        src_attn: 多头注意力对象Q!=K=V
        feed_farward: 前馈前连接层
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_farward
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        memory: 语义存储变量
        source_mask: 源数据掩码张量
        target_mask: 目标数据掩码张量
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, source_mask))

        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):

        for layer in self.layers:
            x = layer(x, lambda: layer(x, memory, src_mask, target_mask))

        return self.norm(x)


# 将线性层 + softmax计算层
class Generator(nn.Module):
    def __init__(self, d_model: int, vocal_size: int):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocal_size)

    def forward(self, x):
        x = self.project(x)
        return F.log_softmax(x, dim=-1)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, target, src_mask, target_mask):
        memory = self.encode(source, src_mask)
        return self.decode(memory, src_mask, target, target_mask)

    def encode(self, source, src_mask):
        x = self.src_embed(source)
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        x = self.target_embed(target)
        return self.decoder(x, memory, src_mask, target_mask)


def make_model(
    source_vocab,
    target_vocab,
    N=6,
    d_model=512,
    d_ff=2048,
    head=8,
    max_len=4096,
    dropout=0.1,
):
    # source_vocab: 源数据词汇总数
    # target_vocab: 代表目标数据的词汇总数
    # N: 代表编码器和解码器堆叠的层数
    # d_model: 代表词嵌入的维度
    # d_ff: 前馈前连接层变换矩阵的维度
    # head: 多头注意力的头数
    # dropout
    dcp = copy.deepcopy
    attn = MutiheaderAttention(head, d_model)
    feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    pe = PositionalEncoding(d_model, p_dropout=dropout, max_len=max_len)
    encoder = Encoder(EncoderLayer(d_model, dcp(attn), dcp(feed_forward), dropout), N)
    decoder = Decoder(
        DecoderLayer(
            size=d_model,
            self_attn=dcp(attn),
            src_attn=dcp(attn),
            feed_farward=dcp(feed_forward),
            dropout=dropout,
        ),
        N,
    )
    generator = Generator(d_model=d_model, vocal_size=target_vocab)
    model = EncoderDecoder(
        encoder,
        decoder,
        src_embed=nn.Sequential(
            Embedding(embedding_dim=d_model, num_embeddings=source_vocab), dcp(pe)
        ),
        target_embed=nn.Sequential(
            Embedding(embedding_dim=d_model, num_embeddings=target_vocab), dcp(pe)
        ),
        generator=generator,
    )
    # 初始化整个模型中的参数， 判断参数的维度大于1，将矩阵初始化为一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


print(make_model(source_vocab=11, target_vocab=11))

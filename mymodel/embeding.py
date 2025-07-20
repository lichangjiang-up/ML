import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embed  = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        self.sqrt_embedding_dim = math.sqrt(embedding_dim)
    
    def forward(self, x):
        """该函数为向前传播逻辑，所有层都有。
        当传给该类对象实例化参数时，自动调用该函数。
        参数x: embedding为首层, x为输入token下标张量"""
        return self.embed(x) * self.sqrt_embedding_dim



class PositionEncoding(nn.Module):
    def __init__(self, embedding_dim, p_dropout, max_len):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_dropout)

        # 初始化一个位置编码0矩阵Size(max_len, embedding_dim)
        pe = torch.zeros(max_len, embedding_dim)

        # 初始化一个绝对索引矩阵，
        position = torch.arange(0, max_len).unsqueeze(1)

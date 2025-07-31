import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np 
from torch.autograd import Variable

from mymodel.embeding import Embedding, PositionalEncoding


def test_embedding():
    embedding_dim = 6
    num_embeddings = 10
    max_len = 6
    embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    ipt = Variable(
        torch.LongTensor([[5, 3, 0, 5, 4, 7], [6, 4, 8, 8, 9, 0], [0, 1, 2, 6, 7, 0]])
    )
    embedings = embed(ipt)
    # print(embedings)
    embed = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    embedings = embed(ipt)
    pe = PositionalEncoding(embedding_dim=embedding_dim, max_len=max_len, p_dropout=.2)
    pe_embedings = pe(embedings)
    print("pe_embedings", pe_embedings)
    plt.plot(pe_embedings[0, :, :].data.numpy())
    # plt.legend()
    plt.show()




def test_arrange():
    max_len = 50
    embedding_dim = 20
    position = torch.arange(0, max_len).unsqueeze(1)
    pe = torch.zeros(max_len, embedding_dim)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-1) / embedding_dim)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # print("test_arrange", pe.unsqueeze(0))

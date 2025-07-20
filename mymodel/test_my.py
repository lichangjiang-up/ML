import torch.nn as nn 
import torch
from torch.autograd import Variable

from mymodel.embeding import Embedding


class test_embedding:
    embed = nn.Embedding(num_embeddings=10, embedding_dim=5)    
    ipt = Variable(torch.LongTensor([[5, 3, 0, 5, 4, 7 ], [6, 4, 8, 8, 9, 0], [0, 1, 2, 6, 7, 0]]))
    embedings = embed(ipt)
    print(embedings)
    embed = nn.Embedding(num_embeddings=10, embedding_dim=5, padding_idx=0) 
    embedings = embed(ipt)
    print(embedings)
    embed = Embedding(num_embeddings=10, embedding_dim=5) 
    embedings = embed(ipt)
    print(embedings, embedings.shape)
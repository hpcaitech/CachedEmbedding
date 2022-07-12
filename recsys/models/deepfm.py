import torch
import torch.nn as nn
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from recsys import ParallelMode
from recsys.modules.embeddings import ParallelQREmbedding, ParallelMixVocabEmbeddingBag


class FeatureEmbedding(nn.Module):
    
    def __init__(self, field_dims, emb_dim, enable_qr):
        super().__init__()
        if enable_qr:
            self.embedding = ParallelQREmbedding(emb_dim, sum(field_dims) // 50, verbose=False)
        else:
            self.embedding = ParallelMixVocabEmbeddingBag(field_dims, emb_dim, mode='mean')

    def forward(self,x):
        return self.embedding(x)
    

class FeatureLinear(nn.Module):

    def __init__(self, field_dims, enable_qr, output_dim=1):
        super().__init__()
        if enable_qr:
            self.fc = ParallelQREmbedding(output_dim, sum(field_dims) // 50, verbose=False)
        else:
            # self.fc = nn.EmbeddingBag(sum(field_dims), output_dim, mode='mean')
            self.fc = ParallelMixVocabEmbeddingBag(field_dims, output_dim, mode='mean')
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        
    def forward(self,x):
        return self.fc(x) + self.bias


class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum
        
    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square

        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):

    def __init__(self, emb_dims, dropout, output_layer=True):
        super().__init__()
        layers = []

        for i in range(len(emb_dims)-1):
            layers.append(nn.Linear(emb_dims[i],emb_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        if output_layer:
            layers.append(nn.Linear(emb_dims[-1],1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        with record_function('MLP layer'):
            return self.mlp(x)
        
        
class DeepFactorizationMachine(nn.Module):
    
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, enable_qr):
        super().__init__()
        self.enable_qr = enable_qr
        self.linear = FeatureLinear(field_dims, enable_qr)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeatureEmbedding(field_dims, embed_dim, enable_qr)
        self.mlp = MultiLayerPerceptron([embed_dim]+mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        # print('[DEBUG] embed x',embed_x.size()) # [16384, 128]
        # print('[DEBUG] linear',self.linear(x).squeeze(1).size())
        # print('[DEBUG] fm',self.fm(embed_x).size())
        # print('[DEBUG] mlp',self.mlp(embed_x).squeeze(-1).size())
        x = self.linear(x).squeeze(1) + self.fm(embed_x) + self.mlp(embed_x).squeeze(-1)
        return torch.sigmoid(x)

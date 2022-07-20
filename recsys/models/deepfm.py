import math

import torch
import torch.nn as nn
from torch.profiler import record_function

from recsys import ParallelMode, DISTLogger, DISTMGR
from recsys.modules.embeddings import ParallelMixVocabEmbeddingBag
from colo_recsys.utils import count_parameters


class FeatureEmbedding(nn.Module):
    
    def __init__(self, field_dims, emb_dim, enable_qr):
        super().__init__()
        self.embedding = ParallelMixVocabEmbeddingBag(field_dims, emb_dim, mode='mean',
                                                          parallel_mode=ParallelMode.TENSOR_PARALLEL,
                                                          enable_qr=enable_qr, do_fair=True)
            
        # print('Saved params (M)',emb_dim*(sum(field_dims) - math.ceil(math.sqrt(sum(field_dims))))//1_000_000)

    def forward(self,sparse_features):
        return self.embedding(sparse_features)
    

class FeatureLinear(nn.Module):

    def __init__(self, dense_input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(dense_input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        
    def forward(self,x):
        return self.linear(x) + self.bias


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
    
    def __init__(self, num_embed_per_feature, dense_input_dim, embed_dim, mlp_dims, dropout, enable_qr):
        super().__init__()
        world_size = DISTMGR.get_world_size()
        rank = DISTMGR.get_rank()
        self.linear = FeatureLinear(dense_input_dim, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeatureEmbedding(num_embed_per_feature, embed_dim, enable_qr)
        self.mlp = MultiLayerPerceptron([embed_dim*2]+mlp_dims, dropout)
    
    def forward(self, sparse_feats, dense_feats):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(sparse_feats)
        linear_x = self.linear(dense_feats)
        combined_x = torch.cat([embed_x, linear_x], dim=1)
        x = self.fm(embed_x) + self.mlp(combined_x).squeeze(-1)
        return torch.sigmoid(x)

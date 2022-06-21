from typing import List

import torch
import torch.nn as nn
import numpy as np

class PEPEmbeddingBag(nn.Module):
    def __init__(self, 
                 field_dims: List[int], 
                 embed_dim: int, 
                 threshold_type: str ='feature_dim'):
        super().__init__()
        self.use_cuda = True
        self.threshold_type = threshold_type
        self.latent_dim = embed_dim
        self.field_dims = field_dims
        self.feature_num = sum(field_dims) 
        self.field_num = len(field_dims)
        self.gk = 1 #args['gk']
        init = -150 #args['threshold_init']

        self.g = torch.sigmoid
        self.s = self.init_threshold(init)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)

        self.v = torch.nn.Parameter(torch.rand(self.feature_num, self.latent_dim))
        torch.nn.init.xavier_uniform_(self.v)

        self.sparse_v = self.v.data


    def init_threshold(self, init):
        if self.threshold_type == 'global':
            s = nn.Parameter(init * torch.ones(1))
        elif self.threshold_type == 'dimension':
            s = nn.Parameter(init * torch.ones([self.latent_dim]))
        elif self.threshold_type == 'feature':
            s = nn.Parameter(init * torch.ones([self.feature_num, 1]))
        elif self.threshold_type == 'field':
            s = nn.Parameter(init * torch.ones([self.field_num, 1]))
        elif self.threshold_type == 'feature_dim':
            s = nn.Parameter(init * torch.ones([self.feature_num, self.latent_dim]))
        elif self.threshold_type == 'field_dim':
            s = nn.Parameter(init * torch.ones([self.field_num, self.latent_dim]))
        else:
            raise ValueError('Invalid threshold_type: {}'.format(self.threshold_type))
        return s

    def soft_threshold(self, v, s):
        if s.size(0) == self.field_num:  # field-wise lambda
            field_v = torch.split(v, tuple(self.field_dims))
            concat_v = []
            for i, v in enumerate(field_v):
                v = torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s[i]) * self.gk))
                concat_v.append(v)

            concat_v = torch.cat(concat_v, dim=0)
            return concat_v
        else:
            return torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s) * self.gk))

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        self.sparse_v = self.soft_threshold(self.v, self.s)
        xv = nn.Embedding(x, self.sparse_v)

        return torch.sum(xv, dim=1)
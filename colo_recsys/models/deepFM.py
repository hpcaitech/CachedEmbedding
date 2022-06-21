import torch
import torch.nn as nn
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import colossalai.nn as col_nn

from recsys.modules.embeddings import QREmbedding

class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims, emb_dim, enable_qr):
        super().__init__()
        if enable_qr:
            self.embedding = QREmbedding(emb_dim, sum(field_dims) // 50, verbose=False)
        else:
            self.embedding = nn.Embedding(sum(field_dims), emb_dim)
        self.offsets = np.array((0,*np.cumsum(field_dims)[:-1]),dtype=np.long)

    def forward(self,x):
        with record_function('Embedding lookup'):
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return self.embedding(x)

    @property
    def weight(self):
        return self.embedding.weight

class FeatureLinear(nn.Module):
    def __init__(self, field_dims, enable_qr, output_dim=1):
        super().__init__()
        # May change to use embeddingbag(mode='sum')
        # self.fc = nn.EmbeddingBag(sum(field_dims), output_dim, mode='sum')
        if enable_qr:
            self.fc = QREmbedding(output_dim, sum(field_dims) // 50, verbose=False)
        else:
            self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.offsets = np.array((0,*np.cumsum(field_dims[:-1])),dtype=np.long)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        
    def forward(self,x):
        with record_function('Linear layer lookup'):
            x = x + x.new_tensor(self.offsets).unsqueeze(0)
            return torch.sum(self.fc(x), dim=1) + self.bias

    @property
    def weight(self):
        return self.fc.weight

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
            # layers.append(col_nn.LayerNorm(emb_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        if output_layer:
            layers.append(nn.Linear(emb_dims[-1],1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        with record_function('MLP layer'):
            return self.mlp(x)

    @property
    def weight(self):
        return [layer.weight for layer in self.mlp]
        
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
        # print('### embed x',embed_x.size()) # [16384, 128]
        # print('### linear',self.linear(x).squeeze(1).size())
        # print('### fm',self.fm(embed_x).size())
        # print('### mlp',torch.sum(self.mlp(embed_x).squeeze(-1),dim=1).size())
        if self.enable_qr:
            x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x).squeeze(1)
        else:
            x = self.linear(x).squeeze(1) + torch.sum(self.fm(embed_x),dim=1) + torch.sum(self.mlp(embed_x).squeeze(-1),dim=1)
        return torch.sigmoid(x)


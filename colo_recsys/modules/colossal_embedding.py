import numpy as np
import torch
import torch.nn as nn

import colossalai
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils.cuda import get_current_device


class EmbeddingCollection(nn.Embedding):

    def __init__(self, num_embeddings_per_feature, embedding_dim):
        tot_features = sum(num_embeddings_per_feature)
        tp_size = gpc.tensor_parallel_size
        if tot_features % tp_size != 0:
            tot_features += (tp_size - tot_features % tp_size)
        super().__init__(tot_features, embedding_dim, sparse=True)

        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        self.register_buffer('offsets', torch.from_numpy(offsets).unsqueeze(0).requires_grad_(False), False)

    def forward(self, x):
        embedding = super().forward(x + self.offsets)
        if self.offsets.device.type == 'cpu':
            embedding = embedding.cuda()
        return embedding

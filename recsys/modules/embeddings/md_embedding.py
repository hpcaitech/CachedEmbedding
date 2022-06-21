from typing import Callable, List

import torch
import torch.nn as nn
import numpy as np


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class MixDimensionEmbeddingBag(nn.Module):
    def __init__(self, 
                 blocks_vocab: List[torch.Tensor], 
                 field_dims: List[int], 
                 blocks_embedding_dims: List[int], 
                 base_embedding_dim: int):

        super().__init__()
        self.num_blocks = len(blocks_vocab)
        self.block_indicies = {}

        for (i, vocab) in enumerate(blocks_vocab):
            for v in vocab:
                self.block_indicies[v] = i

        self.offsets = [np.array((0,*np.cumsum(field_dims[block])[:-1]),dtype=np.long) for block in blocks_vocab]

        self.base_embedding_dim = base_embedding_dim

        self.block_embedding_encoders = nn.ModuleList()
        self.block_embedding_projectors = nn.ModuleList()

        for idx in range(self.num_blocks):
            embedding_dim = blocks_embedding_dims[idx]
            self.block_embedding_encoders.append(
                nn.Embedding(sum(vocab), embedding_dim)
            )
            if embedding_dim == base_embedding_dim:
                self.block_embedding_projectors.append(LambdaLayer(lambda x: x))
            else:
                self.block_embedding_projectors.append(
                    nn.Linear(embedding_dim,base_embedding_dim)
                )
       
        self._init_weights()

    def forward(self, x):
        # x_cpu = x.to('cpu')
        embeddings = torch.zeros(x.size(0), self.base_embedding_dim).to(x.device)
        for idx in range(self.num_blocks):
            ## TODO: move following operations to dataloader
            offset = self.offsets[idx]
            bool_x = [self.block_indicies[y] == idx for y in range(x.size(1))]
            inv_bool_x = [self.block_indicies[y] != idx for y in range(x.size(1))]
            
            with torch.no_grad():
                x_copy = x.copy()
                for row in x_copy:#x_cpu:
                    row[bool_x] = row[bool_x] + torch.tensor(offset)#.to(x.device)
                    row[inv_bool_x] = 0
            
            # x = x_cpu.to(x.device)
            
            encoder = self.block_embedding_encoders[idx].to(x.device)
            
            block_embeddings = encoder(x_copy)
            projector = self.block_embedding_projectors[idx].to(x.device)
            block_embeddings = projector(block_embeddings)
            # FIX: [cuda error] device-side assert triggered (might be with embedding lookup)
            # print('## mask',torch.tensor(bool_x).view(1,len(bool_x),1))
            mask = torch.tensor(bool_x).view(1,len(bool_x),1).to(x.device)
            block_embeddings = block_embeddings * mask
            block_embeddings = torch.sum(block_embeddings, dim=1)
            
            embeddings = embeddings + block_embeddings

        return embeddings

    def _init_weights(self):
        for embed in self.block_embedding_encoders:
            nn.init.xavier_uniform_(embed.weight.data)
        for proj in self.block_embedding_projectors:
            nn.init.xavier_uniform_(proj.weight.data)
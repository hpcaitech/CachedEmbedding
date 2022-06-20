import torch
import torch.nn as nn


class QREmbedding(nn.Module):
    def __init__(self, embedding_dim, num_buckets, verbose):
        super().__init__()
        self.num_buckets = num_buckets
        self.q_embeddings = nn.Embedding(num_buckets, embedding_dim,)
        self.r_embeddings = nn.Embedding(num_buckets, embedding_dim,)
        self.verbose = verbose
        self._init_weights()

    def forward(self, x, offsets=None):
        if offsets is not None:
            x = x + x.new_tensor(self.offsets).unsqueeze(0)

        # Get the quotient index.
        if self.verbose:
            print('## input x:',x)
        
        quotient_index = torch.floor_divide(x, self.num_buckets)
        
        # Get the reminder index.
        remainder_index = torch.remainder(x, self.num_buckets)
        
        if self.verbose:
            print('## Q-index:',quotient_index.size())
            print('## R-index:',remainder_index.size())
        
        # Lookup the quotient_embedding using the quotient_index.
        quotient_embedding = self.q_embeddings(quotient_index)
        
        # Lookup the remainder_embedding using the remainder_index.
        remainder_embedding = self.r_embeddings(remainder_index)
        
        if self.verbose:
            print('## Q-embedding:',quotient_embedding.size())
            print('## R-embedding:',remainder_embedding.size())
        
        # Use multiplication as a combiner operation
        return torch.sum(quotient_embedding * remainder_embedding, dim=1)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_embeddings.weight.data)
        nn.init.xavier_uniform_(self.r_embeddings.weight.data)

    @property
    def weight(self):
        return self.q_embeddings.weight + self.r_embeddings.weight
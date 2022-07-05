import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from recsys import DISTMGR, ParallelMode, DISTLogger
from ..functional import reduce_forward


class LoadBalanceManager(object):
    def __init__(self, field_dims: List[int], num_groups=4, base_emb_dim=128):
        assert len(field_dims) >= num_groups, \
                f"number of input fields {len(field_dims)} must be larger than the world size {num_groups}"
        self.field_dims = field_dims
        self.num_groups = num_groups
        self.base_emb_dim = base_emb_dim
        self._initialize()
    
    def _initialize(self) -> None:
        dim_indices = np.array(range(len(self.field_dims)))
        np.random.shuffle(dim_indices)
        chunk_size = len(self.field_dims) // self.num_groups
        self.groups = []

        for i in range(self.num_groups):
            if i == self.num_groups-1:
                self.groups.append(dim_indices[i*chunk_size:])
                break
            self.groups.append(dim_indices[i*chunk_size:(i+1)*chunk_size])

        self.emb_dims = []
        total_sum = sum(self.field_dims)
        for group in self.groups:
            # scale base embedding dim by total_sum/sum
            div = total_sum / sum([self.field_dims[x] for x in group])
            # divide base dim by a 2^n nearest to div
            emb_dim = int(self.base_emb_dim / 2**(int(math.log2(div))))
            self.emb_dims.append(emb_dim)

    def get_group(self, rank: int) -> List[List[int]]:
        assert hasattr(self, 'groups') and rank in range(0, self.num_groups)
        return self.groups[rank]

    def get_block_dim(self, rank: int) -> List[int]:
        assert hasattr(self, 'emb_dims') and rank in range(0, self.num_groups)
        return self.emb_dims[rank]

    def get_field_dims(self) -> List[int]:
        assert hasattr(self, 'field_dims') and self.field_dims is not None
        return self.field_dims

    def get_base_dim(self) -> int:
        assert hasattr(self, 'base_emb_dim') and self.base_emb_dim is not None
        return self.base_emb_dim

    def shard_tensor(self, _input: Tensor, rank: int) -> Tensor:
        assert hasattr(self, 'groups')
        group = self.groups[rank]
        assert min(group) >= 0 and max(group) < _input.size(1)
        return _input[:, group]


class BlockEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 block_embedding_dim: int = 64,
                 base_embedding_dim: int = 128,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: int = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 mode: str = 'sum',
                 include_last_offset: bool = False,
                 embed_w: Optional[Tensor] = None,
                 linear_w: Optional[Tensor] = None,
                 freeze_w: Optional[bool] = False,
                 device = None,
                 dtype = None,
                 init_method = nn.init.xavier_normal_):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.block_embedding_dim = block_embedding_dim
        self.base_embedding_dim = base_embedding_dim
        
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Specific to embedding bag
        self.mode = mode
        self.include_last_offset = include_last_offset
        
        if embed_w is None:
            self.embed_weight = nn.Parameter(
                torch.empty(num_embeddings, block_embedding_dim, device=device, dtype=dtype))
            if padding_idx is not None:
                with torch.no_grad():
                    self.embed_weight[padding_idx].fill_(0)
            init_method(self.embed_weight)
        else:
            assert list(embed_w.shape) == [num_embeddings, block_embedding_dim]
            self.embed_weight = nn.Parameter(embed_w, requires_grad=(not freeze_w))

        if block_embedding_dim == base_embedding_dim:
            self.linear_weight = None
        else:
            if linear_w is None:
                self.linear_weight = nn.Parameter(
                    torch.empty(base_embedding_dim, block_embedding_dim, device=device, dtype=dtype))
                init_method(self.linear_weight)
            else:
                assert list(linear_w.shape) == [base_embedding_dim, block_embedding_dim], \
                    "Pretrained weights have dimension {x1}, which is different from linear layer dimensions {x2} \
                        ".format(x1=list(linear_w.shape), x2=[block_embedding_dim, base_embedding_dim])
                self.linear_weight = nn.Parameter(linear_w, requires_grad=(not freeze_w))
           
    def forward(self, input_, offsets=None, per_sample_weights=None):
        output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)
        if self.block_embedding_dim != self.base_embedding_dim:
            output_parallel = F.linear(output_parallel, self.linear_weight, bias=None)
        
        assert output_parallel.size() == (input_.size(0), self.base_embedding_dim)
        return output_parallel

    @classmethod
    def from_pretrained(cls,
                        weights: List[Tensor],
                        base_embedding_dim: int = 128,
                        freeze: bool = True,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        mode: str = 'sum',
                        sparse: bool = False,
                        include_last_offset: bool = False,
                        padding_idx: Optional[int] = None,
                        init_method = nn.init.xavier_normal_) -> 'BlockEmbeddingBag':
        assert len(weights) == 2 and weights[0].dim() == 2, \
            'Both embedding and linear weights are expected \n \
            Embedding parameters are expected to be 2-dimensional'
        rows, cols = weights[0].shape
        embeddingbag = cls(num_embeddings=rows,
                           block_embedding_dim=cols,
                           base_embedding_dim=base_embedding_dim,
                           embed_w=weights[0],
                           linear_w=weights[1],
                           freeze_w=freeze,
                           max_norm=max_norm,
                           norm_type=norm_type,
                           scale_grad_by_freq=scale_grad_by_freq,
                           mode=mode,
                           sparse=sparse,
                           include_last_offset=include_last_offset,
                           padding_idx=padding_idx,
                           init_method=init_method)
        return embeddingbag

    def get_base_embedding_dim(self) -> int:
        assert hasattr(self, 'base_embedding_dim')
        return self.base_embedding_dim

    def get_weights(self, detach: bool = False) -> List[Optional[Tensor]]:
        assert isinstance(self.embed_weight, Tensor)
        if self.linear_weight is None:
            return [self.embed_weight.detach() if detach 
                    else self.embed_weight, None]
        else:
            assert isinstance(self.linear_weight, Tensor)
            return [self.embed_weight.detach() if detach 
                    else self.embed_weight, self.linear_weight]


class ParallelMixVocabEmbeddingBag(nn.Module):

    def __init__(self,
                field_dims: List[int],
                embedding_dim: int = 128,
                parallel_mode: Optional[ParallelMode] = None,
                mode: str = 'sum',
                blk_embed: Optional[BlockEmbeddingBag] = None,
                lbmgr: Optional[LoadBalanceManager] = None,
                freeze: bool = False,
                *args,
                **kwargs):
        super().__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.mode = mode

        # Decide number of nodes
        self.parallel_mode = ParallelMode.DEFAULT if parallel_mode is None else parallel_mode
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.num_groups = self.world_size # default setting

        if lbmgr is not None:
            self.lbmgr = lbmgr
            self.field_dims = lbmgr.get_field_dims()
            self.embedding_dim = lbmgr.get_base_dim()
        else:
            self.lbmgr = LoadBalanceManager(field_dims, self.num_groups, embedding_dim)

        self.group = self.lbmgr.get_group(self.rank)
        self.block_dim = self.lbmgr.get_block_dim(self.rank)
        self.comm_func = reduce_forward

        if blk_embed is not None:
            weights = blk_embed.get_weights(detach=True)
            base_embedding_dim = blk_embed.get_base_embedding_dim()
            assert weights[0].size() == (sum([self.field_dims[i] for i in self.group]), self.block_dim), \
                'passed embedding layer dimensions are wrong: {x1} vs {x2} \
                    '.format(x1=weights[0].size(), x2=(sum([self.field_dims[i] for i in self.group]), self.block_dim))
            if self.block_dim != self.embedding_dim:
                assert weights[1].size() == (self.embedding_dim, self.block_dim), \
                    'passed linear layer dimensions are wrong: {x1} vs {x2} \
                    '.format(x1=weights[1].size(), x2=(self.embedding_dim, self.block_dim))
            if base_embedding_dim != self.embedding_dim:
                DISTLogger.warning('Base embedding dimension provided by blk_embed is different from \
                    default or manually passed. Will overwrite by blk_embed.base_embedding_dim')
                self.embedding_dim = base_embedding_dim
            self.embed = BlockEmbeddingBag.from_pretrained(
                                                weights=weights,
                                                base_embedding_dim=base_embedding_dim,
                                                freeze=freeze)
        else:
            self.embed = BlockEmbeddingBag(
                                    sum([self.field_dims[i] for i in self.group]), 
                                    block_embedding_dim=self.block_dim,
                                    base_embedding_dim=self.embedding_dim,
                                    mode=mode,
                                    *args,
                                    **kwargs)

    def forward(self, x, offsets=None):
        x_parallel = self.lbmgr.shard_tensor(x, self.rank)
        output_parallel = self.embed(x_parallel, offsets)
        output_gather = self.comm_func(output_parallel, self.parallel_mode)#, reduce_op=self.mode)

        if self.mode == 'mean':
            with torch.no_grad():
                output_gather = output_gather / self.num_groups

        assert output_gather.shape == (x.size(0), self.embedding_dim)
        return output_gather

    @classmethod
    def from_pretrained(cls,
                        blk_embed: BlockEmbeddingBag,
                        lbmgr: LoadBalanceManager,
                        mode: str = 'sum',
                        freeze: bool = False,
                        field_dims: Optional[List[int]] = None,
                        embedding_dim: Optional[int] = 128,
                        parallel_mode: Optional[ParallelMode] = None,
                        *args,
                        **kwargs) -> 'ParallelMixVocabEmbeddingBag':
        assert not (field_dims is None and blk_embed is None),\
             'field_dims and blk_embed cannot both be None'
        assert not (field_dims is None and lbmgr is None), \
            'field_dims and load balance manager cannot both be None'
        embeddingbag = cls(
                    field_dims=field_dims,
                    embedding_dim=embedding_dim,
                    parallel_mode=parallel_mode,
                    mode=mode,
                    blk_embed=blk_embed,
                    lbmgr=lbmgr,
                    freeze=freeze,
                    *args,
                    **kwargs)

        return embeddingbag
    
    def get_weights(self, detach: bool = False) -> List[Tensor]:
        assert hasattr(self, 'embed') and isinstance(self.embed, BlockEmbeddingBag)
        return self.embed.get_weights(detach)
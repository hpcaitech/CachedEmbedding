import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ReduceOp
import numpy as np

from colossalai.utils import get_current_device
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from colossalai.communication import all_reduce

np.random.seed(123)  
_reduce_ops = dict(sum=ReduceOp.SUM, max=ReduceOp.MAX, mean=ReduceOp.SUM)
logger = get_dist_logger()


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
        # dim_indices = np.argsort(self.field_dims)
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
            emb_dim = max(2, int(self.base_emb_dim / 2**(int(math.log2(div)))))
            self.emb_dims.append(emb_dim)
            
    def shard_tensor(self, _input: Tensor, rank: int, device = None) -> Tensor:
        assert hasattr(self, 'groups') and rank in range(0, self.num_groups)
        group = self.groups[rank]
        assert min(group) >= 0 and max(group) < _input.size(1)
        return _input[:, group].to(device)
            
    def get_group(self, rank: int) -> List[List[int]]:
        assert hasattr(self, 'groups') and rank in range(0, self.num_groups)
        return self.groups[rank]

    def get_block_dim(self, rank: int) -> List[int]:
        assert rank in range(0, self.num_groups)
        return self.emb_dims[rank]

    def get_field_dims(self) -> List[int]:
        return self.field_dims

    def get_base_dim(self) -> int:
        return self.base_emb_dim
    
    def get_qr_bucket_size(self, rank: int) -> int:
        group = self.get_group(rank)
        group_sum = sum([self.field_dims[x] for x in group])
        qr_bucket_size = math.ceil(math.sqrt(group_sum))
        return qr_bucket_size


class QREmbeddingBag(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 num_buckets: int,
                 embedding_dim: int = 128,
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: int = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 mode: str = 'sum',
                 include_last_offset: bool = False,
                 embed_ws: Optional[List[Tensor]] = None,
                 freeze_w: Optional[bool] = False,
                 device = None,
                 dtype = None,
                 init_method = nn.init.xavier_normal_):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        
        print('Params savings {:.3f}B'.format((num_embeddings*embedding_dim - 2*num_buckets*embedding_dim)/1_000_000_000))

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
            self.q_padding_idx = torch.div(padding_idx, self.num_buckets, rounding_mode='floor')
            self.r_padding_idx = torch.remainder(padding_idx, self.num_buckets)
        else:
            self.q_padding_idx = self.r_padding_idx = None
        
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Specific to embedding bag
        self.mode = mode
        self.include_last_offset = include_last_offset
        
        if embed_ws is None:
            self.quotient_embed_weight = nn.Parameter(
                torch.empty(self.num_buckets, embedding_dim, device=device, dtype=dtype))
            self.remainder_embed_weight = nn.Parameter(
                torch.empty(self.num_buckets, embedding_dim, device=device, dtype=dtype))
            if padding_idx is not None:
                with torch.no_grad():
                    self.quotient_embed_weight[self.q_padding_idx].fill_(0)
                    self.remainder_embed_weight[self.r_padding_idx].fill_(0)
            init_method(self.quotient_embed_weight)
            init_method(self.remainder_embed_weight)
        else:
            assert list(embed_ws[0].shape) == [self.num_buckets, embedding_dim]
            assert list(embed_ws[1].shape) == [self.num_buckets, embedding_dim]
            self.quotient_embed_weight = nn.Parameter(embed_ws[0], requires_grad=(not freeze_w))
            self.remainder_embed_weight = nn.Parameter(embed_ws[1], requires_grad=(not freeze_w))
            
    def forward(self, input_, offsets=None, per_sample_weights=None):
        # Get the quotient index.
        quotient_ids = torch.div(input_, self.num_buckets, rounding_mode='floor')
        # Get the reminder index.
        remainder_ids = torch.remainder(input_, self.num_buckets)
        # Lookup the quotient_embedding using the quotient_index.
        quotient_embed = F.embedding_bag(quotient_ids, self.quotient_embed_weight, offsets, self.max_norm, 
                                             self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, 
                                             per_sample_weights, self.include_last_offset, self.q_padding_idx) # Q-embedding
        # Lookup the remainder_embedding using the remainder_index.
        remainder_embed = F.embedding_bag(remainder_ids, self.remainder_embed_weight, offsets, self.max_norm, 
                                             self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, 
                                             per_sample_weights, self.include_last_offset, self.r_padding_idx) # R-embedding

        # Use multiplication as a combiner operation
        output_parallel = quotient_embed + remainder_embed
        assert output_parallel.size() == (input_.size(0), self.embedding_dim)
        return output_parallel

    @classmethod
    def from_pretrained(cls,
                        weights: List[Tensor],
                        num_embeddings: int,
                        freeze: bool = True,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        mode: str = 'sum',
                        sparse: bool = False,
                        include_last_offset: bool = False,
                        padding_idx: Optional[int] = None,
                        init_method = nn.init.xavier_normal_) -> 'BlockEmbeddingBag':
        assert len(weights) == 2 and weights[0].dim() == 2 and weights[1].dim() == 2, \
            'Both embedding weights are expected to be 2-dimensional'
        num_buckets, embedding_dim = weights[0].shape
        embeddingbag = cls(num_embeddings=num_embeddings,
                           num_buckets=num_buckets,
                           embedding_dim=embedding_dim,
                           embed_ws=weights,
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

    def get_num_embeddings(self) -> int:
        return self.num_embeddings

    def get_weights(self, detach: bool = False) -> List[Optional[Tensor]]:
        assert isinstance(self.quotient_embed_weight, Tensor)
        assert isinstance(self.remainder_embed_weight, Tensor)
        return [self.quotient_embed_weight.detach() if detach else self.quotient_embed_weight, \
                self.remainder_embed_weight.detach() if detach else self.remainder_embed_weight]
            

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
        
        print(self.block_embedding_dim,self.base_embedding_dim)
        
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
                pretrain_embed: Optional[Union[BlockEmbeddingBag,QREmbeddingBag]] = None,
                enable_qr: bool = False,
                lbmgr: Optional[LoadBalanceManager] = None,
                freeze: bool = False,
                device = None,
                *args,
                **kwargs):
        super().__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.device = device if device is not None else get_current_device()

        # Decide number of nodes
        self.parallel_mode = ParallelMode.GLOBAL if parallel_mode is None else parallel_mode
        self.world_size = gpc.get_world_size(self.parallel_mode)
        self.rank = gpc.get_local_rank(self.parallel_mode)
        self.num_groups = self.world_size # default setting

        if lbmgr is not None:
            self.lbmgr = lbmgr
            self.field_dims = lbmgr.get_field_dims()
            self.embedding_dim = lbmgr.get_base_dim()
        else:
            self.lbmgr = LoadBalanceManager(field_dims, self.num_groups, embedding_dim)

        self.group = self.lbmgr.get_group(self.rank)
        self.block_dim = self.lbmgr.get_block_dim(self.rank)
        self.qr_bucket_size = self.lbmgr.get_qr_bucket_size(self.rank)
        self.offsets = torch.tensor((0,*np.cumsum(np.array( \
            self.field_dims, dtype=np.long)[self.group])[:-1]), device=self.device)
        
        self.comm_func = all_reduce
        cls = BlockEmbeddingBag if not enable_qr else QREmbeddingBag

        if pretrain_embed is not None:
            assert isinstance(pretrain_embed, cls), f"{type(pretrain_embed)} isn't {cls}"
            if not enable_qr:
                weights = pretrain_embed.get_weights(detach=True)
                base_embedding_dim = pretrain_embed.get_base_embedding_dim()
                assert weights[0].size() == (sum([self.field_dims[i] for i in self.group]), self.block_dim), \
                    'passed embedding layer dimensions are wrong: {x1} vs {x2} \
                        '.format(x1=weights[0].size(), x2=(sum([self.field_dims[i] for i in self.group]), self.block_dim))
                if self.block_dim != self.embedding_dim:
                    assert weights[1].size() == (self.embedding_dim, self.block_dim), \
                        'passed linear layer dimensions are wrong: {x1} vs {x2} \
                        '.format(x1=weights[1].size(), x2=(self.embedding_dim, self.block_dim))
                if base_embedding_dim != self.embedding_dim:
                    logger.warning('Base embedding dimension provided by blk_embed is different from \
                        default or manually passed. Will overwrite by blk_embed.base_embedding_dim')
                    self.embedding_dim = base_embedding_dim
                self.embed = cls.from_pretrained(weights=weights,
                                            base_embedding_dim=base_embedding_dim,
                                            freeze=freeze)
            else:
                weights = pretrain_embed.get_weights(detach=True)
                num_embeddings = pretrain_embed.get_num_embeddings()
                assert num_embeddings == sum([self.field_dims[i] for i in self.group]), \
                    f'passed embedding layer have wrong number of embeddings: {num_embeddings} vs \
                        {sum([self.field_dims[i] for i in self.group])}'
                assert weights[0].size() == weights[1].size() == (self.qr_bucket_size, self.embedding_dim), \
                    'either q- or r-embedding layer has wrong input dimensions: {x1} vs {x2} vs {x3} \
                        '.format(x1=weights[0].size(), x2=weights[0].size(), 
                                x3=(self.qr_bucket_size, self.embedding_dim))
                self.embed = cls.from_pretrained(weights=weights,
                                            num_embeddings=num_embeddings,
                                            freeze=freeze)
        else:
            if not enable_qr:
                self.embed = cls(sum([self.field_dims[i] for i in self.group]), 
                                block_embedding_dim=self.block_dim,
                                base_embedding_dim=self.embedding_dim,
                                mode=mode,
                                *args,
                                **kwargs)
            else:
                self.embed = cls(sum([self.field_dims[i] for i in self.group]), 
                                num_buckets=self.qr_bucket_size,
                                embedding_dim=self.embedding_dim,
                                mode=mode,
                                *args,
                                **kwargs)

    def _shard_tensor(self, _input: Tensor) -> Tensor:
        assert min(self.group) >= 0 and max(self.group) < _input.size(1)
        return _input[:, self.group].to(self.device)
    
    def forward(self, x):
        x_parallel = self._shard_tensor(x)
        x_parallel = x_parallel + self.offsets
        output_parallel = self.embed(x_parallel)
        output_gather = self.comm_func(output_parallel, self.parallel_mode, op=_reduce_ops[self.mode])

        if self.mode == 'mean':
            output_gather = output_gather / self.num_groups

        assert output_gather.shape == (x.size(0), self.embedding_dim)
        return output_gather

    @classmethod
    def from_pretrained(cls,
                        pretrain_embed: Union[BlockEmbeddingBag, QREmbeddingBag],
                        lbmgr: LoadBalanceManager,
                        enable_qr: bool = False,
                        mode: str = 'sum',
                        freeze: bool = False,
                        field_dims: Optional[List[int]] = None,
                        embedding_dim: Optional[int] = 128,
                        parallel_mode: Optional[ParallelMode] = None,
                        *args,
                        **kwargs) -> 'ParallelMixVocabEmbeddingBag':
        assert not (field_dims is None and lbmgr is None), \
            'field_dims and load balance manager cannot both be None'
        embeddingbag = cls(
                    field_dims=field_dims,
                    embedding_dim=embedding_dim,
                    parallel_mode=parallel_mode,
                    mode=mode,
                    pretrain_embed=pretrain_embed,
                    enable_qr=enable_qr,
                    lbmgr=lbmgr,
                    freeze=freeze,
                    *args,
                    **kwargs)

        return embeddingbag
    
    def get_weights(self, detach: bool = False) -> List[Tensor]:
        assert hasattr(self, 'embed') and isinstance(self.embed, (BlockEmbeddingBag, QREmbeddingBag))
        return self.embed.get_weights(detach)
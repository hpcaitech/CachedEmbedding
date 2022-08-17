from collections import defaultdict
import math
from functools import partial
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import Tensor
import numpy as np

from recsys import DISTMGR, ParallelMode, DISTLogger
from ..functional import reduce_forward
from .load_balance_mgr import LoadBalanceManager

np.random.seed(111)  
REDUCE_OPS = dict(max=lambda x,dim:torch.max(x,dim=dim)[0], mean=torch.mean, sum=torch.sum)


class QREmbeddingBag(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 num_embeddings: int,
                 qr_bucket_size: int,
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
        self.qr_bucket_size = qr_bucket_size
        self.embedding_dim = embedding_dim
        
        print('Params savings {:.3f}B'.format((num_embeddings*embedding_dim - 2*qr_bucket_size*embedding_dim)/1_000_000_000))

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
            self.q_padding_idx = torch.div(padding_idx, self.qr_bucket_size, rounding_mode='floor')
            self.r_padding_idx = torch.remainder(padding_idx, self.qr_bucket_size)
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
                torch.empty(self.qr_bucket_size, embedding_dim, device=device, dtype=dtype))
            self.remainder_embed_weight = nn.Parameter(
                torch.empty(self.qr_bucket_size, embedding_dim, device=device, dtype=dtype))
            if padding_idx is not None:
                with torch.no_grad():
                    self.quotient_embed_weight[self.q_padding_idx].fill_(0)
                    self.remainder_embed_weight[self.r_padding_idx].fill_(0)
            init_method(self.quotient_embed_weight)
            init_method(self.remainder_embed_weight)
        else:
            assert list(embed_ws[0].shape) == [self.qr_bucket_size, embedding_dim]
            assert list(embed_ws[1].shape) == [self.qr_bucket_size, embedding_dim]
            self.quotient_embed_weight = nn.Parameter(embed_ws[0], requires_grad=(not freeze_w))
            self.remainder_embed_weight = nn.Parameter(embed_ws[1], requires_grad=(not freeze_w))
            
    def forward(self, input_, offsets=None, per_sample_weights=None):
        # Get the quotient index.
        quotient_ids = torch.div(input_, self.qr_bucket_size, rounding_mode='floor')
        # Get the reminder index.
        remainder_ids = torch.remainder(input_, self.qr_bucket_size)
        # Lookup the quotient_embedding using the quotient_index.
        quotient_embed = F.embedding_bag(quotient_ids, self.quotient_embed_weight, offsets, self.max_norm, 
                                             self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, 
                                             per_sample_weights, self.include_last_offset, self.q_padding_idx) # Q-embedding
        # Lookup the remainder_embedding using the remainder_index.
        remainder_embed = F.embedding_bag(remainder_ids, self.remainder_embed_weight, offsets, self.max_norm, 
                                             self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse, 
                                             per_sample_weights, self.include_last_offset, self.r_padding_idx) # R-embedding

        # Use multiplication as a combiner operation
        output_parallel = quotient_embed * remainder_embed
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
        qr_bucket_size, embedding_dim = weights[0].shape
        embeddingbag = cls(num_embeddings=num_embeddings,
                           qr_bucket_size=qr_bucket_size,
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

    def get_num_embeddings_on_rank(self) -> int:
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
        self.device = device
        self.dtype = dtype
        
        print('Saved params (M)',(self.num_embeddings*(self.base_embedding_dim-self.block_embedding_dim)\
                                - self.block_embedding_dim*self.base_embedding_dim)/1_000_000)

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
        self.freeze_w = freeze_w

        # Specific to embedding bag
        self.mode = mode
        self.include_last_offset = include_last_offset

        if embed_w is None:
            self.embed_weight = nn.Parameter(
                torch.empty(num_embeddings, block_embedding_dim, device=self.device, dtype=dtype))
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.embed_weight[self.padding_idx].fill_(0)
            init_method(self.embed_weight)
        else:
            assert list(embed_w.shape) == [num_embeddings, block_embedding_dim]
            self.embed_weight = nn.Parameter(embed_w, 
                                             requires_grad=(not self.freeze_w)).to(self.device)

        if block_embedding_dim == base_embedding_dim:
            self.linear_weight = None
        else:
            if linear_w is None:
                self.linear_weight = nn.Parameter(
                    torch.empty(base_embedding_dim, block_embedding_dim, device=self.device, dtype=dtype))
                init_method(self.linear_weight)
            else:
                assert list(linear_w.shape) == [base_embedding_dim, block_embedding_dim], \
                    "Pretrained weights have dimension {x1}, which is different from linear layer dimensions {x2} \
                        ".format(x1=list(linear_w.shape), x2=[block_embedding_dim, base_embedding_dim])
                self.linear_weight = nn.Parameter(linear_w, 
                                                  requires_grad=(not self.freeze_w)).to(self.device)

    def forward(self, input_: Tensor, offsets=None, per_sample_weights=None):
        input_, nonzero_counts_ = self._handle_unexpected_inputs(input_)
        # if self.mode == 'mean':
        #     output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
        #                                     self.scale_grad_by_freq, 'sum', self.sparse, per_sample_weights,
        #                                     self.include_last_offset, self.padding_idx)
            # if nonzero_counts_ is not None:
            #     output_parallel /= nonzero_counts_.unsqueeze(-1)
        # else:
        output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
                                        self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                        self.include_last_offset, self.padding_idx)

        if self.block_embedding_dim != self.base_embedding_dim:
            output_parallel = F.linear(output_parallel, self.linear_weight, bias=None)
        assert output_parallel.size() == (input_.size(0), self.base_embedding_dim)
        return output_parallel
    
    def _handle_unexpected_inputs(self, input_: Tensor) -> Tensor:
        nonzero_counts_ = None
        if torch.max(input_) > self.num_embeddings or torch.min(input_) < 0:
            if self.padding_idx is None:
                self.padding_idx = self.num_embeddings
                _embed_weight = torch.empty((self.num_embeddings+1, self.block_embedding_dim),
                                            device=self.device, dtype=self.dtype)
                _padding_weight = torch.zeros((1, self.block_embedding_dim),
                                            device=self.device, dtype=self.dtype)
                _embed_weight.data.copy_(torch.cat([self.embed_weight.data,
                                                   _padding_weight.data],
                                                   dim=0))
                self.embed_weight = nn.Parameter(_embed_weight)
            with torch.no_grad():
                self.embed_weight[self.padding_idx].fill_(-float('inf') 
                                                            if self.mode == 'max' else 0)
            # if self.mode == 'mean':
            #     nonzero_counts_ = torch.max(
            #         torch.sum((input_ < self.num_embeddings) & (input_ >= 0), dim=1),
            #         torch.ones(input_.size(0),device=input_.device))
            input_[(input_ >= self.num_embeddings) | (input_ < 0)] = self.padding_idx
        return input_, nonzero_counts_

    @classmethod
    def from_pretrained(cls,
                        weights: List[Tensor],
                        base_embedding_dim: int = 128,
                        freeze: bool = False,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        mode: str = 'sum',
                        sparse: bool = False,
                        include_last_offset: bool = False,
                        padding_idx: Optional[int] = None,
                        device = None,
                        dtype = None,
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
                           device=device,
                           dtype=dtype,
                           init_method=init_method)
        return embeddingbag

    def get_base_embedding_dim(self) -> int:
        return self.base_embedding_dim

    def get_weights(self, detach: bool = False) -> List[Optional[Tensor]]:
        assert isinstance(self.embed_weight, Tensor)
        if detach and self.padding_idx is not None and \
            self.padding_idx == self.num_embeddings:
            self.embed_weight = nn.Parameter(self.embed_weight[:self.padding_idx,:],
                                             requires_grad=(not self.freeze_w))
        if self.linear_weight is None:
            return [self.embed_weight.detach() if detach 
                    else self.embed_weight, None]
        else:
            return [self.embed_weight.detach() if detach 
                    else self.embed_weight, self.linear_weight.detach() if detach else self.linear_weight]


def determine_freq_blocks(word_frequencies: Tensor,
                          num_embeddings: int,
                          num_blocks: int,
                          base_embedding_dim: int) \
    -> Tuple[List[int], Dict, List[int]]:
    assert word_frequencies.dim()==1 and len(word_frequencies) == num_embeddings
    dim_indices = np.argsort(word_frequencies.numpy())[::-1]
    sort_word_freqs = np.sort(word_frequencies.numpy())[::-1]
    tile_len = num_embeddings // num_blocks
    total_sum = sum(word_frequencies)
    block_embedding_dims = [0] * num_blocks
    num_embeddings_per_block = [0] * num_blocks
    id_group_map = defaultdict(dict) # group-id -> {feature id}
    
    def compute_block_dim(quotient):
        return max(2, int(base_embedding_dim / 2**(int(math.log2(quotient)))))

    for i in range(num_blocks):
        if i == num_blocks-1:
            local_sum = sum(sort_word_freqs[i*tile_len:])
            indices = dim_indices[i*tile_len:]
        else:
            local_sum = sum(sort_word_freqs[i*tile_len:(i+1)*tile_len])
            indices = dim_indices[i*tile_len:(i+1)*tile_len]
        block_dim = compute_block_dim(total_sum / local_sum)
        in_group_index = 0
        for j in indices:
            id_group_map[i][j] = in_group_index
            in_group_index += 1
        block_embedding_dims[i] = block_dim
        num_embeddings_per_block[i] = len(indices)
        
    return num_embeddings_per_block, id_group_map, block_embedding_dims

class MultiBlockEmbeddingBag(nn.Module):
    def __init__(self,
                 word_frequencies: Tensor,
                 num_embeddings: int,
                 num_blocks: int = 4,
                 base_embedding_dim: int = 128,
                 mode: str = 'sum', 
                 device = None,
                 *args,
                 **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_embeddings_per_block , self.id_group_map, self.block_embedding_dims = \
                determine_freq_blocks(word_frequencies,num_embeddings,self.num_blocks,base_embedding_dim)
        self.base_embedding_dim = base_embedding_dim
        self._sanity_check()
        self.block_embeds = [BlockEmbeddingBag(
                                self.num_embeddings_per_block[i],
                                self.block_embedding_dims[i],
                                self.base_embedding_dim,
                                device=device,
                                mode=mode,
                                *args,
                                **kwargs) 
                             for i in range(self.num_blocks)]
        self.mode = mode
        self.device = device
        
    def _sanity_check(self):
        assert self.num_blocks>=1 and self.num_blocks == len(self.num_embeddings_per_block)
        assert self.num_blocks == len(self.block_embedding_dims)
        assert self.base_embedding_dim >= max(self.block_embedding_dims)
        
    def _get_input_on_block(self, input_: Tensor, block: int) -> List[Tensor]:
        required_ids = set(self.id_group_map[block].keys())
        mask = torch.logical_or([torch.eq(input_,torch.ones(input_.shape,device=input_.device)*id) 
                                 for id in required_ids],dim=0)
        return input_[mask]
        
    def forward(self, input_: Tensor, offsets=None, per_sample_weights=None):
        assert input_.dim() == 2
        outputs = []
        for i in range(self.num_blocks):
            cinput_ = self._get_input_on_block(input_, i)
            outputs.append(self.block_embeds[i](cinput_, offsets, per_sample_weights).unsqueeze(0))
        return REDUCE_OPS[self.mode](torch.cat(outputs,dim=0),dim=0)


class ParallelMixVocabEmbeddingBag(nn.Module):

    def __init__(self,
                embeddings_per_feat: List[int],
                embedding_dim: int = 128,
                parallel_mode: Optional[ParallelMode] = None,
                mode: str = 'sum',
                pretrain_embed: Optional[Union[BlockEmbeddingBag,QREmbeddingBag]] = None,
                enable_qr: bool = False,
                lbmgr: Optional[LoadBalanceManager] = None,
                freeze: bool = False,
                device = None,
                do_fair: bool = True,
                *args,
                **kwargs):
        super().__init__()
        self.embeddings_per_feat = embeddings_per_feat
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.device = device if device is not None else torch.device('cuda', torch.cuda.current_device())
        self._do_fair = do_fair

        # Decide number of nodes
        self.parallel_mode = ParallelMode.DEFAULT if parallel_mode is None else parallel_mode
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.num_groups = self.world_size # default setting

        if lbmgr is not None:
            self._lbmgr = lbmgr
            self._do_fair = do_fair
            self.embeddings_per_feat = lbmgr.get_embeddings_per_feat()
            self.embedding_dim = lbmgr.get_base_dim()
        else:
            self._lbmgr = LoadBalanceManager(embeddings_per_feat, self.num_groups,\
                embedding_dim, do_fair=self._do_fair, device=self.device)

        self.num_embeddings_on_rank = self._lbmgr.get_num_embeddings_on_rank(self.rank)
        self.block_dim = self._lbmgr.get_block_dim(self.rank)
        self.qr_bucket_size = self._lbmgr.get_qr_bucket_size(self.rank)

        self.comm_func = reduce_forward
        cls = BlockEmbeddingBag if not enable_qr else QREmbeddingBag

        if pretrain_embed is not None:
            assert isinstance(pretrain_embed, cls), f"{type(pretrain_embed)} isn't {cls}"
            if not enable_qr:
                weights = pretrain_embed.get_weights(detach=True)
                base_embedding_dim = pretrain_embed.get_base_embedding_dim()
                assert weights[0].size() == (self.num_embeddings_on_rank, self.block_dim), \
                    'passed embedding layer dimensions are wrong: {x1} vs {x2} \
                        '.format(x1=weights[0].size(), x2=(self.num_embeddings_on_rank, self.block_dim))
                if self.block_dim != self.embedding_dim:
                    assert weights[1].size() == (self.embedding_dim, self.block_dim), \
                        'passed linear layer dimensions are wrong: {x1} vs {x2} \
                        '.format(x1=weights[1].size(), x2=(self.embedding_dim, self.block_dim))
                if base_embedding_dim != self.embedding_dim:
                    DISTLogger.warning('Base embedding dimension provided by blk_embed is different from \
                        default or manually passed. Will overwrite by blk_embed.base_embedding_dim')
                    self.embedding_dim = base_embedding_dim
                self.embed = cls.from_pretrained(weights=weights,
                                            base_embedding_dim=base_embedding_dim,
                                            freeze=freeze)
            else:
                weights = pretrain_embed.get_weights(detach=True)
                num_embeddings = pretrain_embed.get_num_embeddings_on_rank()
                assert num_embeddings == self.num_embeddings_on_rank, \
                    f'passed embedding layer have wrong number of embeddings: {num_embeddings} vs \
                        {self.num_embeddings_on_rank}'
                assert weights[0].size() == weights[1].size() == (self.qr_bucket_size, self.embedding_dim), \
                    'either q- or r-embedding layer has wrong input dimensions: {x1} vs {x2} vs {x3} \
                        '.format(x1=weights[0].size(), x2=weights[0].size(), 
                                x3=(self.qr_bucket_size, self.embedding_dim))
                self.embed = cls.from_pretrained(weights=weights,
                                            num_embeddings=num_embeddings,
                                            freeze=freeze)
        else:
            if not enable_qr:
                # unify linear weight initialization
                linear_weight = torch.randn((self.embedding_dim,self.block_dim),device=self.device)
                self.embed = cls(self.num_embeddings_on_rank, 
                                block_embedding_dim=self.block_dim,
                                base_embedding_dim=self.embedding_dim,
                                mode=mode,
                                device=self.device,
                                linear_w=linear_weight,
                                *args,
                                **kwargs)
            else:
                self.embed = cls(self.num_embeddings_on_rank, 
                                qr_bucket_size=self.qr_bucket_size,
                                embedding_dim=self.embedding_dim,
                                mode=mode,
                                device=self.device,
                                *args,
                                **kwargs)
    
    def forward(self, x, offsets=None):
        assert offsets is None, "offsets have been handled internally.\n Users shouldn't manually input offsets"
        x_parallel = self._lbmgr.shard_tensor(x, self.rank)
        output_parallel = self.embed(x_parallel)
        output_gather = self.comm_func(output_parallel, self.parallel_mode, reduce_op=self.mode)
        assert output_gather.shape == (x.size(0), self.embedding_dim)
        return output_gather 

    @classmethod
    def from_pretrained(cls,
                        pretrain_embed: Union[BlockEmbeddingBag, QREmbeddingBag],
                        lbmgr: LoadBalanceManager,
                        enable_qr: bool = False,
                        mode: str = 'sum',
                        freeze: bool = False,
                        embeddings_per_feat: Optional[List[int]] = None,
                        embedding_dim: Optional[int] = 128,
                        parallel_mode: Optional[ParallelMode] = None,
                        *args,
                        **kwargs) -> 'ParallelMixVocabEmbeddingBag':
        assert pretrain_embed is not None, 'pretrained embedding weights are required'
        assert not (embeddings_per_feat is None and lbmgr is None), \
            'embeddings_per_feat and load balance manager cannot both be None'
        embeddingbag = cls(
                    embeddings_per_feat=embeddings_per_feat,
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
    
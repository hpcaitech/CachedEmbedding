import math
from collections import defaultdict
from typing import Dict, List, Optional, Union
from sympy import numbered_symbols

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from recsys import DISTMGR, ParallelMode, DISTLogger
from ..functional import reduce_forward

np.random.seed(111)  
REDUCE_OPS = dict(min=lambda x,dim:torch.min(x,dim=dim)[0], 
                  max=lambda x,dim:torch.max(x,dim=dim)[0], 
                  mean=torch.mean, sum=torch.sum)


class LoadBalanceManager(object):
    def __init__(self, embeddings_per_feat: List[int], num_groups=4, base_emb_dim=128, \
        do_fair=True, device=None, disable_random_behavior=False):
        assert len(embeddings_per_feat) >= num_groups, \
                f"number of input fields {len(embeddings_per_feat)} must be larger than the world size {num_groups}"
        self.embeddings_per_feat = embeddings_per_feat
        self.num_groups = num_groups
        self.base_emb_dim = base_emb_dim
        self.do_fair = do_fair
        self.device = device
        self.all_feat_offsets = torch.cumsum(torch.tensor([0]+self.embeddings_per_feat,
                                                          device=self.device),dim=0)
        if not self.do_fair:
            self._shuffle_initialize(disable_random_behavior)
        else:
            self._fair_initialize()
        
    def _fair_initialize(self) -> None:
        self.num_embeddings_per_rank = sum(self.embeddings_per_feat) // self.num_groups
        dim_indices = np.array(range(len(self.embeddings_per_feat)))
        self.groups = []
        self.offsets = []
        _curr_grp = []
        _curr_offs = [0]

        self.cuts = dict()
        _num_cuts = 0
        _agg = self.num_embeddings_per_rank
        # Find cut positions and shard groups
        for ind in dim_indices:
            while self.embeddings_per_feat[ind] > _agg:
                if _num_cuts >= self.num_groups - 1: # never cut when enough groups
                    break
                if ind in self.cuts.keys():
                    self.cuts[ind].append(_agg)
                else:
                    self.cuts[ind] = [_agg]
                _num_cuts += 1
                
                self.offsets.append(torch.from_numpy(np.asarray(_curr_offs,dtype=np.int64)).to(self.device))
                _curr_offs = [0]
                _curr_grp.append(ind)
                self.groups.append(_curr_grp)
                _curr_grp = []
                
                _agg += self.num_embeddings_per_rank
            
            if _agg >= self.embeddings_per_feat[ind] and len(_curr_offs) == 1:
                _curr_offs.append(self.embeddings_per_feat[ind]-(_agg-self.num_embeddings_per_rank))
            else:
                _curr_offs.append(self.embeddings_per_feat[ind])
            
            _agg -= self.embeddings_per_feat[ind]
            _curr_grp.append(ind)
        
        self.offsets.append(torch.from_numpy(np.asarray(_curr_offs[:-1],dtype=np.int64)).to(self.device))
        for i in range(len(self.offsets)):
            self.offsets[i] = torch.cumsum(self.offsets[i], dim=0)
            
        self.groups.append(_curr_grp)
        
        self.emb_dim = max(2, int(self.base_emb_dim / 
                                  2**(int(math.log2(self.num_groups)))))
        self.qr_bucket_size = math.ceil(math.sqrt(self.num_embeddings_per_rank))
        
    def _shuffle_initialize(self, disable_random_behavior=False) -> None:
        if disable_random_behavior:
            dim_indices = np.argsort(self.embeddings_per_feat)
        else:
            dim_indices = np.array(range(len(self.embeddings_per_feat)))
            np.random.shuffle(dim_indices)
        self.dim_indices = dim_indices
        chunk_size = len(self.embeddings_per_feat) // self.num_groups
        self.groups = []

        for i in range(self.num_groups):
            if i == self.num_groups-1:
                self.groups.append(dim_indices[i*chunk_size:])
                break
            self.groups.append(dim_indices[i*chunk_size:(i+1)*chunk_size])

        self.emb_dims = []
        total_sum = sum(self.embeddings_per_feat)
        for group in self.groups:
            div = total_sum / sum([self.embeddings_per_feat[x] for x in group])
            emb_dim = max(2, int(self.base_emb_dim / 2**(int(math.log2(div)))))
            self.emb_dims.append(emb_dim)
            
        self.qr_bucket_sizes = [math.ceil(math.sqrt(sum([self.embeddings_per_feat[x] for x in group]))) 
                               for group in self.groups]
        
        self.offsets = [torch.tensor((0,*np.cumsum(np.array( \
                            self.embeddings_per_feat, dtype=np.int64)[group])[:-1]), device=self.device)
                        for group in self.groups]

    def get_group(self, rank: int) -> List[int]:
        assert rank in range(0, self.num_groups)
        return list(self.groups[rank])
    
    def get_offsets(self, rank: int = 0, return_all: bool=False) -> List[int]:
        if return_all:
            return self.all_feat_offsets[:-1]
        assert rank in range(0, self.num_groups)
        return self.offsets[rank]
        
    def get_num_embeddings_on_rank(self, rank: int) -> int:
        if not self.do_fair:
            group = self.get_group(rank)
            return sum([self.embeddings_per_feat[i] for i in group])
        else:
            return self.num_embeddings_per_rank
    
    def get_block_dim(self, rank: int) -> int:
        assert rank in range(0, self.num_groups)
        if not self.do_fair:
            return self.emb_dims[rank]
        else:
            return self.emb_dim
    
    def get_qr_bucket_size(self, rank: int) -> int:
        """deprecated as qr embedding is no longer of use"""
        if not self.do_fair:
            assert rank in range(len(self.qr_bucket_sizes))
            return self.qr_bucket_sizes[rank]
        else:
            return self.qr_bucket_size
        
    def shard_tensor_deprecated(self, _input: Tensor, rank: int) -> Tensor:
        assert _input.dim() == 2 and _input.size(1) == len(self.embeddings_per_feat)
        offsets = self.get_offsets(rank)
        if not self.do_fair:
            group = self.get_group(rank)
            assert min(group) >= 0 and max(group) < _input.size(1)
            return _input[:, group] + offsets
        else:
            if self.num_groups == 1: # no cut
                assert rank == 0, 'Maximum rank exceeded, no cut is performed'
                return _input + offsets
            _cinput = _input.clone()
            feats = list(self.cuts.keys())
            assert hasattr(self, 'cuts'), 'lbmgr object has no cuts attribute'
            assert rank in range(self.num_groups), 'invalid rank'
            if rank == 0:
                feat_id = feats[0]
                cut_pos = self.cuts[feat_id][0]
                _cinput[:,feat_id] = torch.min((cut_pos-offsets[-1]-1)*torch.ones(_cinput.size(0),device=self.device),\
                                                _cinput[:,feat_id])
                return _cinput[:,:feat_id+1] + offsets 
            else:
                rank -= 1
                for (k,v) in self.cuts.items():
                    if rank - len(v) < 0:
                        cut_pos = v[rank:][:2] # this and next shard position
                        if len(cut_pos) < 2 and k != feats[-1]:
                            next_feat_id = feats[feats.index(k) + 1]
                        else:
                            next_feat_id = None
                        feat_id = k
                        break
                    rank -= len(v)
                if len(cut_pos) == 1:
                    cut_pos = cut_pos[0]
                    if feat_id == feats[-1]: # last rank
                        _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=self.device),
                                                       _cinput[:,feat_id]-cut_pos)
                        return _cinput[:,feat_id:] + offsets
                    else:
                        assert next_feat_id is not None
                        _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=self.device), \
                                                       _cinput[:,feat_id]-cut_pos)
                        _cinput[:,next_feat_id] = torch.min((cut_pos-offsets[-1]-1)*torch.ones(_cinput.size(0), \
                                                                    device=self.device),_cinput[:,next_feat_id])
                        return _cinput[:,feat_id:next_feat_id+1] + offsets
                elif len(cut_pos) == 2:
                    pos1, pos2 = cut_pos
                    _cinput[:,feat_id] = torch.max(torch.zeros(_cinput.size(0),device=self.device), 
                                                    _cinput[:,feat_id]-pos1)
                    _cinput[:,feat_id] = torch.min((pos2-pos1-offsets[-1]-1)*torch.ones(_cinput.size(0),device=self.device), 
                                                        _cinput[:,feat_id])
                    return _cinput[:,feat_id:feat_id+1] + offsets
                else:
                    raise ValueError('input tensor and embeddings_per_feat do not match. Double check inputs.')
         
    def faster_shard_tensor_deprecated(self, _input: Tensor, rank:int) -> Tensor:
        """simpler and more correct shard_tensor function"""
        assert self.do_fair, 'offset only supports fair division'
        assert _input.dim() == 2 and _input.size(1) == len(self.embeddings_per_feat)
        if self.device is not None:
            assert _input.device == self.device, 'input device {x1} should be consistent with lbmgr device {x2}'\
                                .format(x1=_input.device,x2=self.device)
        group = self.get_group(rank)
        _cinput = _input[:,group].clone() + self.all_feat_offsets[group]
        assert torch.max(_cinput) < self.all_feat_offsets[-1], \
            'input tensor should not have offsets added beforehand'
        num_embeddings_this_rank = self.get_num_embeddings_on_rank(rank)
        lower_bnd = rank * num_embeddings_this_rank
        # shard input tensor into expected interval
        _cinput = _cinput - lower_bnd
        return _cinput.to(torch.int64)

    def shard_tensor(self, _input: Tensor, rank:int, use_deprecated:bool=False) -> Tensor:
        """old interface compatible"""
        if use_deprecated:
            return self.faster_shard_tensor_deprecated(_input, rank)
        else:
            return self.put_tensor_on_rank(_input, rank)
    
    def put_tensor_on_rank(self, _input: Tensor, rank: int) -> Tensor:
        assert self.do_fair, 'currently only fair initialization is supported'
        # add offsets
        assert _input.dim() == 2 and _input.size(1) == len(self.embeddings_per_feat)
        if self.device is not None:
            assert _input.device == self.device, 'input device {x1} should be consistent with lbmgr device {x2}'\
                                .format(x1=_input.device,x2=self.device)   
        num_embeddings_this_rank = self.get_num_embeddings_on_rank(rank)
        lower_bnd = rank * num_embeddings_this_rank
        _cinput = _input.clone() + self.all_feat_offsets[:-1] - lower_bnd
        assert _cinput.shape == (_input.size(0), len(self.embeddings_per_feat))
        return _cinput
         
    def shard_weights(self, weights: Tensor, rank: int) -> Tensor:
        if weights is None:
            return weights
        assert weights.dim() == 2 and weights.size(0) == self.all_feat_offsets[-1]
        if not self.do_fair:
            group = self.get_group(rank)
            shard_weights = []
            for i in range(1,len(self.embeddings_per_feat)):
                if i in group:
                    shard_weights.append(weights[self.all_feat_offsets[i-1]:
                        self.all_feat_offsets[i],:])
            return torch.cat(shard_weights, dim=0)
        else:
            num_embeddings = self.get_num_embeddings_on_rank(rank)
            if rank == self.num_groups - 1:
                return weights[num_embeddings*rank:,:]
            return weights[num_embeddings*rank:num_embeddings*(rank+1),:]

    def get_embeddings_per_feat(self) -> List[int]:
        return self.embeddings_per_feat

    def get_base_dim(self) -> int:
        return self.base_emb_dim


class QREmbeddingBag(nn.Module):
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
        assert self.mode in REDUCE_OPS
        
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
                                             requires_grad=(not self.freeze_w),
                                             device=self.device)

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
                                                  requires_grad=(not self.freeze_w),
                                                  device=self.device)

    def forward(self, input_: Tensor, offsets=None, per_sample_weights=None):
        input_ = self.handle_unexpected_inputs(input_)
        output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)
        if self.block_embedding_dim != self.base_embedding_dim:
            output_parallel = F.linear(output_parallel, self.linear_weight, bias=None)
            
        assert output_parallel.size() == (input_.size(0), self.base_embedding_dim)
        return output_parallel
    
    def handle_unexpected_inputs(self, input_: Tensor) -> Tensor:
        """function to handle input with illegal feature ids"""
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
                    self.embed_weight[self.padding_idx].fill_(0)
            input_[(input_ >= self.num_embeddings) | (input_ < 0)] = self.padding_idx
        return input_

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


class AdaEmbeddingBag(BlockEmbeddingBag):
    def __init__(self, 
                update_word_frequency: bool = True, 
                word_frequencies: Optional[Union[List,Tensor]] = None,
                num_of_quantiles: int = 5,
                update_count: int = 20,
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)
        self.update_word_frequency = update_word_frequency
        self.update_count = self.curr_count = update_count
        self.num_of_quantiles = num_of_quantiles
        if word_frequencies is not None:
            self.register_buffer('word_frequencies',torch.tensor(word_frequencies))
        else:
            self.register_buffer('word_frequencies',torch.zeros(self.num_embeddings))
        self.embed_weight = [self.embed_weight.clone()]
        self._update_weights()
        
    def _padding_zeros(self, _input: Tensor) -> None:
        return F.pad(_input,(0, self.block_embedding_dim-_input.size(0)),'constant',0)
    
    def _update_weights(self) -> None:
        total = sum(self.word_frequencies)
        if total == 0:
            return
        else:
            # TODO: using quantiles and multiprocessing to parallelize embedding lookup of different lengths
            # _embed_weight = torch.empty(self.embed_weight.shape)
            for (i,v) in enumerate(self.word_frequencies):
                _v = max(2,int(self.block_embedding_dim / int(math.log(total / max(1,v)))))
                self.embed_weight[i] = self._padding_zeros(self.embed_weight[i,:_v].clone().detach())
            # self.embed_weight = nn.Parameter(_embed_weight)

    def toggle_update_behavior(self, disable=True):
        assert hasattr(self, 'update_word_frequency')
        if disable:
            self.update_word_frequency = False
        else:
            self.update_word_frequency = True
        
        self.curr_count=self.update_count

    def set_word_frequencies(self, word_frequencies: Union[List,Tensor]):
        assert len(word_frequencies) == self.num_embeddings, \
                        'ensure input word frequencies cover all possible words'
        self.word_frequencies = torch.tensor(word_frequencies.copy())
        self._update_weights()

    def forward(self, input_, offsets=None, per_sample_weights=None):
        if self.update_word_frequency and self.curr_count >= 0:
            for batch in input_:
                self.word_frequencies[batch] += 1
            self.curr_count -= 1
            if self.curr_count < 0:
                self._update_weights()
                self.toggle_update_behavior()

        output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)

        if self.block_embedding_dim != self.base_embedding_dim:
            output_parallel = F.linear(output_parallel, self.linear_weight, bias=None)

        assert output_parallel.size() == (input_.size(0), self.base_embedding_dim)
        return output_parallel


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
                self.embed = cls(self.num_embeddings_on_rank, 
                                block_embedding_dim=self.block_dim,
                                base_embedding_dim=self.embedding_dim,
                                mode=mode,
                                device=self.device,
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
        x_parallel = self._lbmgr.put_tensor_on_rank(x, self.rank)
        output_parallel = self.embed(x_parallel)
        output_gather = self.comm_func(output_parallel, self.parallel_mode, reduce_op=self.mode)

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
    
import math
from typing import List

import torch
from torch import Tensor
import numpy as np


def minimize_groupwise_diff(lst: List[int], num_grp: int) -> List[List[int]]:
    """Compute a greedy solution to group numbers into a desired number of groups
    and minimize the maximum difference between groups. 

    Args:
        lst (List[int]): list of ungrouped numbers
        num_grp (int): number of groups to be formed into

    Returns:
        List[List[int]]: greedily computed grouping of numbers
    """
    if num_grp == 1:
        return [lst]
    indices = list(np.argsort(lst)[::-1])
    mean = sum(lst) // num_grp
    tol = sum(lst) % num_grp
    balance = sum(lst)
    j_pos = len(indices)-1
    groups = []
    for i_ in range(len(indices)):
        if len(groups) + 1 == num_grp: # add residual elements
            groups.append(indices[i_:j_pos+1])
            break
        i = indices[i_]
        if lst[i] > mean or abs(lst[i] - mean) <= tol:
            groups.append([i])
            balance -= lst[i]
        else:
            agg = lst[i]
            grp = [i]
            for j_ in range(j_pos, i_,-1): # add last/smallest elements to group
                j = indices[j_]
                if agg < mean:
                    agg += lst[j]
                    balance -= lst[j]
                    grp.append(j)
                    if balance < mean * (num_grp-len(groups)-1): # check balance still enough for later groups
                        break
                else:
                    break
            if len(grp) > 1:
                groups.append(grp[:-1]) # forgo unused indices
            else:
                groups.append(grp)
            j_pos = j_+1
        mean = balance // (num_grp-len(groups)) # adaptive mean value update for residual groups

    return groups[:num_grp]

class LoadBalanceManager(object):
    """A load manager that divides training loads evenly across tensor parallel 
    embedding ranks.
    """
    def __init__(self, embeddings_per_feat: List[int], num_groups=4, base_emb_dim=128, \
        do_fair=True, device=None, disable_random_behavior=False):
        """initiate the manager with raw feature embeddings that have yet to be sharded.

        Args:
            embeddings_per_feat (List[int]): number of embeddings per sparse feature.
            num_groups (int, optional): number of groups to shard into. Usually world size. Defaults to 4.
            base_emb_dim (int, optional): desired embedding dimension for features. Defaults to 128.
            device (_type_, optional): device where load manager is put. Defaults to None.
            disable_random_behavior (bool, optional): set to `True` to disable feature 
            random shuffling, only applied in table-wise sharding scenario. Defaults to False.
        """
        assert len(embeddings_per_feat) >= num_groups, \
                f"number of input fields {len(embeddings_per_feat)} must be larger than the world size {num_groups}"
        self.embeddings_per_feat = embeddings_per_feat
        self.num_groups = num_groups
        self.base_emb_dim = base_emb_dim
        self.do_fair = do_fair
        self.device = device
        # compute the offsets for all set of features 
        self.all_feat_offsets = torch.cumsum(torch.tensor([0]+self.embeddings_per_feat,
                                                          device=self.device),dim=0)
        if not self.do_fair:
            self._shuffle_initialize(disable_random_behavior)
        else:
            self._fair_initialize()

    def _fair_initialize(self) -> None:
        """shards the features with ...
        """
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
        """_summary_

        Args:
            disable_random_behavior (bool, optional): _description_. Defaults to False.
        """
        if disable_random_behavior:
            self.groups = minimize_groupwise_diff(self.embeddings_per_feat, self.num_groups)
        else:
            dim_indices = np.array(range(len(self.embeddings_per_feat)))
            np.random.shuffle(dim_indices)
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
        if not self.do_fair:
            assert rank in range(len(self.qr_bucket_sizes))
            return self.qr_bucket_sizes[rank]
        else:
            return self.qr_bucket_size
        
    def _shard_tensor(self, _input: Tensor, rank: int) -> Tensor:
        assert _input.dim() == 2 and _input.size(1) == len(self.embeddings_per_feat)
        if self.device is not None:
            assert _input.device == self.device, 'input device {x1} should be consistent with lbmgr device {x2}'\
                                .format(x1=_input.device,x2=self.device)   
        offsets = self.get_offsets(rank)
        if not self.do_fair:
            group = self.get_group(rank)
            assert min(group) >= 0 and max(group) < _input.size(1)
            return _input[:, group] + offsets
        else:
            num_embeddings_this_rank = self.get_num_embeddings_on_rank(rank)
            lower_bnd = rank * num_embeddings_this_rank
            _cinput = _input.clone() + self.all_feat_offsets[:-1] - lower_bnd
            assert _cinput.shape == (_input.size(0), len(self.embeddings_per_feat))
            return _cinput

    def shard_tensor(self, _input: Tensor, rank:int) -> Tensor:
        return self._shard_tensor(_input, rank)

    def _shard_weights(self, weights: Tensor, rank: int) -> Tensor:
        """helper function for unit testing"""
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
        
    def shard_weights(self, weights: Tensor, rank: int) -> Tensor:
        return self._shard_weights(weights, rank)

    def get_embeddings_per_feat(self) -> List[int]:
        return self.embeddings_per_feat

    def get_base_dim(self) -> int:
        return self.base_emb_dim
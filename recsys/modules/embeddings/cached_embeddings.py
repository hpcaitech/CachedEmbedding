import enum
import itertools
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from recsys import DISTMGR, ParallelMode, DISTLogger as logger
from recsys.utils import get_partition
from .base_embeddings import BaseEmbeddingBag
from ..functional import dual_all_to_all


class CacheReplacePolicy(enum.Enum):
    Hash = 0    # "hash"
    LFU = 1    # "least frequently used"
    LRU = 2    # "least recently used"


def find_eligible_positions_hash(miss_ids, hit_ids, cache_indices, cache_states=None):
    """
    hash solution
    """
    positions, rehash_count, write_back_positions = [], 0, []
    for each in miss_ids:
        pos = -1
        for i in itertools.count():
            pos = hash_func(each, cache_indices.shape[0], pos, i)
            if cache_indices[pos] not in hit_ids and pos not in positions:
                rehash_count += i
                positions.append(pos)
                if cache_indices[pos] >= 0:
                    write_back_positions.append(pos)
                break
    positions = torch.tensor(positions, dtype=miss_ids.dtype, device=miss_ids.device)
    write_back_positions = torch.tensor(write_back_positions, dtype=miss_ids.dtype, device=miss_ids.device)
    return positions, write_back_positions, rehash_count


def hash_func(embed_id, N, prev, rehash_count):
    return hash((embed_id, prev, rehash_count)) % N


def find_eligible_positions_lfu(miss_ids, forbidden_list, cache_indices, cache_states):
    """
    LFU cache replacement policy
    """
    positions, write_back_positions = [], []
    sorted_freq, sorted_idx = torch.sort(cache_states)
    for freq, idx in zip(sorted_freq, sorted_idx):
        if cache_indices[idx] in forbidden_list:
            continue
        positions.append(idx)
        if freq >= 0:
            write_back_positions.append(idx)
        if len(positions) == miss_ids.shape[0]:
            break

    positions = torch.tensor(positions, dtype=miss_ids.dtype, device=miss_ids.device)
    write_back_positions = torch.tensor(write_back_positions, dtype=miss_ids.dtype, device=miss_ids.device)
    return positions, write_back_positions, 0


class CachedEmbeddingBag(BaseEmbeddingBag):
    """
    A serial module equal to the vanilla torch.nn.EmbeddingBag
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 cache_sets,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 mode='mean',
                 include_last_offset=False,
                 cache_lines=1,
                 cache_replace_policy=CacheReplacePolicy.LFU,
                 device=None,
                 dtype=None,
                 debug=True):
        super(CachedEmbeddingBag, self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                                 scale_grad_by_freq, sparse, mode, include_last_offset)
        # Specific to cache
        self.cache_sets = cache_sets
        self.cache_lines = cache_lines    # TODO: n-way set associative
        self.cache_replace_policy = cache_replace_policy
        self.debug = debug

        self.num_hits_history = [] if debug else None
        self.num_miss_history = [] if debug else None
        self.num_write_back_history = [] if debug else None

        # real embedding weight
        if _weight is None:
            self.weight = torch.empty(num_embeddings, embedding_dim, device='cpu', dtype=dtype)
            self.init_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            self.weight = _weight

        # embedding cached in GPU for training
        self.cache_weight = nn.Parameter(
            torch.zeros(cache_sets * cache_lines, embedding_dim, device=device, dtype=dtype))
        self.register_buffer("cache_indices",
                             torch.empty(cache_sets * cache_lines, device=device, dtype=torch.long).fill_(-1))

        if cache_replace_policy == CacheReplacePolicy.Hash:
            # just a placeholder
            self.register_buffer("cache_states", torch.zeros(1, device=device, dtype=torch.long), persistent=False)
            self._find_eligible_positions = find_eligible_positions_hash
        else:
            self.register_buffer("cache_states",
                                 torch.empty(cache_sets * cache_lines, device=device, dtype=torch.long).fill_(-1))
            if cache_replace_policy == CacheReplacePolicy.LFU:
                self._find_eligible_positions = find_eligible_positions_lfu
            else:
                raise NotImplementedError("I haven't implemented LRU policy...")

    @torch.no_grad()
    def init_parameters(self):
        self.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        if self.padding_idx is not None:
            self.weight[self.padding_idx].fill_(0)

    def forward(self, indices, offsets=None, per_sample_weights=None):
        # NOTE: we only support 1-dim indices with offsets for storage & communication efficiency
        indices = self.update_cached_embedding(indices)

        embeddings = F.embedding_bag(indices, self.cache_weight, offsets, self.max_norm, self.norm_type,
                                     self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                     self.include_last_offset, self.padding_idx)

        return embeddings

    @torch.no_grad()
    def update_cached_embedding(self, raw_indices: torch.Tensor) -> torch.Tensor:
        """update_cached_embedding
        update the cached embedding weight and returns the indices of raw_indices in the updated cached weight.

        Args:
            raw_indices (torch.Tensor): input ids, represented by their indices in global embedding weight.

        Returns:
            torch.Tensor: Indices of input ids in cached weight, which is a small set of the global embedding weight.
        """
        unique_indices, inverse_indices, counts = torch.unique(raw_indices, return_inverse=True, return_counts=True)
        assert unique_indices.shape[0] <= self.cache_weight.shape[0], \
            f"the num of input ids {unique_indices.shape[0]} is larger than " \
            f"the cache size {self.cache_weight.shape[0]}, please increase the cache size or shrink the batch size"

        unique_miss = unique_indices[torch.isin(unique_indices, self.cache_indices, invert=True)]
        unique_hit = unique_indices[torch.isin(unique_indices, unique_miss, assume_unique=True, invert=True)]

        rehash_count, write_back_count = 0, 0
        if unique_miss.shape[0] > 0:
            positions_miss, write_back_positions, rehash_count = self._find_eligible_positions(
                unique_miss, unique_hit, self.cache_indices, self.cache_states)

            assert positions_miss.shape[0] == unique_miss.shape[0], \
                f"pos: {positions_miss.shape[0]}, unique: {unique_miss.shape[0]}"
            write_back_count = write_back_positions.shape[0]
            self._update_cache_(positions_miss, write_back_positions, unique_miss)
        if self.debug:
            # print(f"miss count: #{unique_miss.shape[0]}, rehash count: #{rehash_count}")
            self.num_miss_history.append(unique_miss.shape[0])
            self.num_hits_history.append(unique_hit.shape[0])
            self.num_write_back_history.append(write_back_count)
        return self._map_indices(unique_indices, inverse_indices, counts)

    def _update_cache_(self, positions_miss, write_back_positions, unique_miss):
        # update cpu embedding: write cached gpu embeddings back to cpu
        write_back_embedding_indices = self.cache_indices[write_back_positions]
        write_back_embeddings = self.cache_weight[write_back_positions].cpu()
        self.weight.data[write_back_embedding_indices] = write_back_embeddings

        # update gpu embedding cache: move missing embeddings from cpu to gpu cache
        new_embeddings = torch.index_select(self.weight, dim=0, index=unique_miss.cpu()).cuda()
        self.cache_weight.data[positions_miss] = new_embeddings

        # update cache indices
        self.cache_indices.data[positions_miss] = unique_miss

        # update (optional) cache states
        if self.cache_replace_policy == CacheReplacePolicy.LFU:
            self.cache_states.data[positions_miss] = -1
        elif self.cache_replace_policy == CacheReplacePolicy.LRU:
            raise NotImplementedError()

    def _map_indices(self, unique_indices, inverse_indices, counts):
        cache_indices = self.cache_indices.tolist()
        local_indices = torch.tensor([cache_indices.index(x.item()) for x in unique_indices],
                                     dtype=unique_indices.dtype,
                                     device=unique_indices.device)
        if self.cache_replace_policy == CacheReplacePolicy.LFU:
            self.cache_states.data[local_indices] += counts
        return local_indices[inverse_indices]

    @torch.no_grad()
    def flush_cache_(self):
        cache_mask = self.cache_indices >= 0
        embed_indices = self.cache_indices[cache_mask].cpu()
        local_indices = torch.nonzero(cache_mask).squeeze(1)
        cache_weight = torch.index_select(self.cache_weight, dim=0, index=local_indices).cpu()
        self.weight.data[embed_indices] = cache_weight

    @classmethod
    def from_pretrained(
        cls,
        embeddings: torch.Tensor,
        cache_sets: int,
        cache_lines=1,
        cache_replace_policy=CacheReplacePolicy.LFU,
        freeze: bool = True,
        max_norm: Optional[float] = None,
        norm_type: float = 2.,
        scale_grad_by_freq: bool = False,
        mode: str = 'mean',
        sparse: bool = False,
        include_last_offset: bool = False,
        padding_idx: Optional[int] = None,
    ) -> 'CachedEmbeddingBag':
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding_bag = cls(rows,
                            cols,
                            cache_sets,
                            padding_idx=padding_idx,
                            max_norm=max_norm,
                            norm_type=norm_type,
                            scale_grad_by_freq=scale_grad_by_freq,
                            sparse=sparse,
                            _weight=embeddings,
                            mode=mode,
                            include_last_offset=include_last_offset,
                            cache_lines=cache_lines,
                            cache_replace_policy=cache_replace_policy)
        embedding_bag.cache_weight.requires_grad = not freeze
        return embedding_bag


class ParallelCachedEmbeddingBag(BaseEmbeddingBag):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 cache_sets,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 mode='mean',
                 include_last_offset=False,
                 cache_lines=1,
                 cache_replace_policy=CacheReplacePolicy.LFU,
                 parallel_mode=ParallelMode.DEFAULT,
                 device=None,
                 dtype=None,
                 debug=True):
        super(ParallelCachedEmbeddingBag,
              self).__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                             sparse, mode, include_last_offset)
        # Specific to cache
        self.cache_sets = cache_sets
        self.cache_lines = cache_lines    # TODO: n-way set associative
        self.cache_replace_policy = cache_replace_policy
        self.debug = debug

        self.num_hits_history = [] if debug else None
        self.num_miss_history = [] if debug else None
        self.num_write_back_history = [] if debug else None

        # Comm settings
        self.parallel_mode = parallel_mode
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)

        self.chunk_start_index, self.chunk_end_index, divisible = get_partition(embedding_dim, self.rank,
                                                                                self.world_size)
        self.embedding_dim_per_partition = self.chunk_end_index - self.chunk_start_index
        if debug:
            logger.info(f"init [{self.chunk_start_index}, {self.chunk_end_index}) chunk of "
                        f"{num_embeddings} x {embedding_dim} in rank {self.rank} / {self.world_size}")
        if _weight is None:
            self.weight = torch.empty(self.num_embeddings, self.embedding_dim_per_partition, device='cpu', dtype=dtype)
            self.init_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            chunk = torch.tensor_split(_weight, self.world_size, 1)[self.rank]
            assert list(chunk.shape) == [num_embeddings, self.embedding_dim_per_partition]
            self.weight = chunk

        # embedding cached in GPU for training
        self.cache_weight = nn.Parameter(
            torch.zeros(cache_sets * cache_lines, self.embedding_dim_per_partition, device=device, dtype=dtype))
        self.register_buffer("cache_indices",
                             torch.empty(cache_sets * cache_lines, device=device, dtype=torch.long).fill_(-1))

        if cache_replace_policy == CacheReplacePolicy.Hash:
            # just a placeholder
            self.register_buffer("cache_states", torch.zeros(1, device=device, dtype=torch.long), persistent=False)
            self._find_eligible_positions = find_eligible_positions_hash
        else:
            self.register_buffer("cache_states",
                                 torch.empty(cache_sets * cache_lines, device=device, dtype=torch.long).fill_(-1))
            if cache_replace_policy == CacheReplacePolicy.LFU:
                self._find_eligible_positions = find_eligible_positions_lfu
            else:
                raise NotImplementedError("I haven't implemented LRU policy...")

    @torch.no_grad()
    def init_parameters(self):
        self.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        if self.padding_idx is not None:
            self.weight[self.padding_idx].fill_(0)

    def forward(self, indices, offsets=None, per_sample_weights=None, shape_hook=None, scatter_dim=0, gather_dim=-1):
        indices = self.update_cached_embedding(indices)

        output_shard = F.embedding_bag(indices, self.cache_weight, offsets, self.max_norm, self.norm_type,
                                       self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                       self.include_last_offset, self.padding_idx)
        if shape_hook is not None:
            output_shard = shape_hook(output_shard)

        # TODO: async communication
        return dual_all_to_all(output_shard, self.parallel_mode, scatter_dim=scatter_dim, gather_dim=gather_dim)

    @torch.no_grad()
    def update_cached_embedding(self, raw_indices):
        with record_function("(zhg) categorize indices"):
            unique_indices, inverse_indices, counts = torch.unique(raw_indices, return_inverse=True, return_counts=True)
            assert unique_indices.shape[0] <= self.cache_weight.shape[0], \
                f"the num of input ids {unique_indices.shape[0]} is larger than " \
                f"the cache size {self.cache_weight.shape[0]}, please increase the cache size or shrink the batch size"

            unique_miss = unique_indices[torch.isin(unique_indices, self.cache_indices, invert=True)]
            unique_hit = unique_indices[torch.isin(unique_indices, unique_miss, assume_unique=True, invert=True)]

        rehash_count, write_back_count = 0, 0
        if unique_miss.shape[0] > 0:
            with record_function("(zhg) find eligible cache idx"):
                positions_miss, write_back_positions, rehash_count = self._find_eligible_positions(
                    unique_miss, unique_hit, self.cache_indices, self.cache_states)

            assert positions_miss.shape[0] == unique_miss.shape[0], \
                f"pos: {positions_miss.shape[0]}, unique: {unique_miss.shape[0]}"
            write_back_count = write_back_positions.shape[0]
            with record_function("(zhg) cache update"):
                self._update_cache_(positions_miss, write_back_positions, unique_miss)

        if self.debug:
            logger.info(
                f"miss count: {unique_miss.shape[0]} / {unique_indices.shape[0]}, "
                f"rehash count: #{rehash_count}, "
                f"write back count: #{write_back_count}",
                ranks=[0])
            self.num_hits_history.append(unique_hit.shape[0])
            self.num_miss_history.append(unique_miss.shape[0])
            self.num_write_back_history.append(write_back_count)
        with record_function("(zhg) embed idx -> cache idx"):
            indices = self._map_indices(unique_indices, inverse_indices, counts)
        return indices

    def _update_cache_(self, positions_miss, write_back_positions, unique_miss):
        # update cpu embedding: write cached gpu embeddings back to cpu
        write_back_embedding_indices = self.cache_indices[write_back_positions]
        write_back_embeddings = self.cache_weight[write_back_positions].cpu()
        self.weight.data[write_back_embedding_indices] = write_back_embeddings

        # update gpu embedding cache: move missing embeddings from cpu to gpu cache
        new_embeddings = torch.index_select(self.weight, dim=0, index=unique_miss.cpu()).cuda()
        self.cache_weight.data[positions_miss] = new_embeddings

        # update cache indices
        self.cache_indices.data[positions_miss] = unique_miss

        # update (optional) cache states
        if self.cache_replace_policy == CacheReplacePolicy.LFU:
            self.cache_states.data[positions_miss] = -1
        elif self.cache_replace_policy == CacheReplacePolicy.LRU:
            raise NotImplementedError()

    def _map_indices(self, unique_indices, inverse_indices, counts):
        cache_indices = self.cache_indices.tolist()
        local_indices = torch.tensor([cache_indices.index(x.item()) for x in unique_indices],
                                     dtype=unique_indices.dtype,
                                     device=unique_indices.device)
        if self.cache_replace_policy == CacheReplacePolicy.LFU:
            self.cache_states.data[local_indices] += counts
        return local_indices[inverse_indices]

    @torch.no_grad()
    def flush_cache_(self):
        cache_mask = self.cache_indices >= 0
        embed_indices = self.cache_indices[cache_mask].cpu()
        local_indices = torch.nonzero(cache_mask).squeeze(1)
        cache_weight = torch.index_select(self.cache_weight, dim=0, index=local_indices).cpu()
        self.weight.data[embed_indices] = cache_weight

    @classmethod
    def from_pretrained(cls,
                        embeddings: torch.Tensor,
                        cache_sets: int,
                        freeze: bool = True,
                        padding_idx: Optional[int] = None,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        sparse: bool = False,
                        mode: str = 'mean',
                        include_last_offset: bool = False,
                        cache_lines=1,
                        cache_replace_policy=CacheReplacePolicy.LFU,
                        parallel_mode=ParallelMode.DEFAULT,
                        debug=True) -> 'ParallelCachedEmbeddingBag':
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding_bag = cls(rows,
                            cols,
                            cache_sets,
                            padding_idx=padding_idx,
                            max_norm=max_norm,
                            norm_type=norm_type,
                            scale_grad_by_freq=scale_grad_by_freq,
                            sparse=sparse,
                            _weight=embeddings,
                            mode=mode,
                            include_last_offset=include_last_offset,
                            cache_lines=cache_lines,
                            cache_replace_policy=cache_replace_policy,
                            parallel_mode=parallel_mode,
                            debug=debug)
        embedding_bag.cache_weight.requires_grad = not freeze
        return embedding_bag

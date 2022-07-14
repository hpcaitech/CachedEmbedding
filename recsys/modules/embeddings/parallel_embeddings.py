# This script is mainly inspired by torch.nn.Embedding & https://github.com/NVIDIA/Megatron-LM
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter

from recsys import DISTMGR, ParallelMode, DISTLogger as logger
from ..functional import (reduce_forward, tensor_gather_forward_split_backward, gather_forward_split_backward,
                          split_forward_gather_backward, dual_all_to_all)

from typing import Optional


def get_vocab_range(num_embeddings, rank, world_size):
    if world_size == 1:
        return 0, num_embeddings

    assert num_embeddings % world_size == 0
    chunk_size = num_embeddings // world_size
    return rank * chunk_size, (rank + 1) * chunk_size


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 _weight=None,
                 device=None,
                 dtype=None,
                 parallel_mode=None,
                 init_method=torch.nn.init.xavier_normal_):    # I suppose this init method aligns with tensorflow?
        super(VocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        # TODO: make sure these options operational
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.parallel_mode = ParallelMode.DEFAULT if parallel_mode is None else parallel_mode
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)

        self.vocab_start_index, self.vocab_end_index = get_vocab_range(num_embeddings, self.rank, self.world_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # padding works
        if padding_idx is not None and self.vocab_start_index <= padding_idx < self.vocab_end_index:
            self.actual_padding_idx = padding_idx - self.vocab_start_index
        else:
            self.actual_padding_idx = None

        if _weight is None:
            self.weight = torch.nn.Parameter(
                torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=device, dtype=dtype))

            # TODO: check RNG states
            init_method(self.weight)
            if self.actual_padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.actual_padding_idx].fill_(0)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            chunk = torch.split(_weight, self.num_embeddings_per_partition, 0)[self.rank]
            self.weight = torch.nn.Parameter(chunk)

    def forward(self, input_):
        if self.world_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            # masked_input = input_.clone() - self.vocab_start_index  # removing clone passes the test
            masked_input = input_ - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight, self.actual_padding_idx, self.max_norm, self.norm_type,
                                      self.scale_grad_by_freq, self.sparse)

        # Mask the output embedding.
        if self.world_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_forward(output_parallel, self.parallel_mode)
        return output


class ColumnParallelEmbeddingBag(torch.nn.Module):
    """EmbeddingBag parallelized in the hidden dimension.

    This version tries best to evenly split the weight parameters even if the column dim is indivisible
    by the world size of the process group
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 mode='mean',
                 include_last_offset=False,
                 _weight=None,
                 device=None,
                 dtype=None,
                 parallel_mode=None,
                 output_device_type=None,
                 init_method=torch.nn.init.xavier_normal_):
        super(ColumnParallelEmbeddingBag, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        # TODO: make sure these options supported by original EmbeddingBag operational
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Specific to embedding bag
        self.mode = mode
        self.include_last_offset = include_last_offset

        # Comm settings
        self.parallel_mode = ParallelMode.DEFAULT if parallel_mode is None else parallel_mode
        self.rank = DISTMGR.get_rank(self.parallel_mode)
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)

        self.chunk_start_index, self.chunk_end_index, divisible = self.get_partition(
            embedding_dim, self.rank, self.world_size)
        self.embedding_dim_per_partition = self.chunk_end_index - self.chunk_start_index
        self.comm_func = gather_forward_split_backward if divisible else tensor_gather_forward_split_backward

        # Init weight
        if _weight is None:
            self.weight = torch.nn.Parameter(
                torch.empty(self.num_embeddings, self.embedding_dim_per_partition, device=device, dtype=dtype))

            # TODO: check RNG states
            with torch.no_grad():
                init_method(self.weight)
                if self.padding_idx is not None:
                    self.weight[self.padding_idx].fill_(0)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            chunk = torch.tensor_split(_weight, self.world_size, 1)[self.rank]
            assert list(chunk.shape) == [num_embeddings, self.embedding_dim_per_partition]
            self.weight = torch.nn.Parameter(chunk)

        self.output_device_type = output_device_type

    @staticmethod
    def get_partition(embedding_dim, rank, world_size):
        if world_size == 1:
            return 0, embedding_dim, True

        assert embedding_dim >= world_size, \
            f"Embedding dimension {embedding_dim} must be larger than the world size " \
            f"{world_size} of the process group"
        chunk_size = embedding_dim // world_size
        threshold = embedding_dim % world_size
        # if embedding dim is divisible by world size
        if threshold == 0:
            return rank * chunk_size, (rank + 1) * chunk_size, True

        logger.warning(
            f"Embedding dimension {embedding_dim} is not divisible by world size {world_size}. "
            f"torch.distributed.all_gather_object introducing additional copy operations is enabled",
            ranks=[0])

        # align with the split strategy of torch.tensor_split
        size_list = [chunk_size + 1 if i < threshold else chunk_size for i in range(world_size)]
        offset = sum(size_list[:rank])
        return offset, offset + size_list[rank], False

    @classmethod
    def from_pretrained(cls,
                        embeddings: Tensor,
                        freeze: bool = True,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        mode: str = 'mean',
                        sparse: bool = False,
                        include_last_offset: bool = False,
                        padding_idx: Optional[int] = None,
                        parallel_mode=None,
                        init_method=torch.nn.init.xavier_normal_) -> 'ColumnParallelEmbeddingBag':
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embeddingbag = cls(num_embeddings=rows,
                           embedding_dim=cols,
                           _weight=embeddings,
                           max_norm=max_norm,
                           norm_type=norm_type,
                           scale_grad_by_freq=scale_grad_by_freq,
                           mode=mode,
                           sparse=sparse,
                           include_last_offset=include_last_offset,
                           padding_idx=padding_idx,
                           parallel_mode=parallel_mode,
                           init_method=init_method)
        embeddingbag.weight.requires_grad = not freeze
        return embeddingbag

    def forward(self, input_, offsets=None, per_sample_weights=None):
        output_parallel = F.embedding_bag(input_, self.weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)
        if self.output_device_type == 'cuda' and output_parallel.device.type == 'cpu':
            # copy-before-transfer instead of transfer-before-copy
            output_parallel = output_parallel.cuda()
        output = self.comm_func(output_parallel, self.parallel_mode, dim=1)
        return output


class FusedHybridParallelEmbeddingBag(ColumnParallelEmbeddingBag):
    """
    For the hybrid parallelism where embedding params use model parallelism
    while the subsequent dense modules use data parallelism

    fused_op:
        - all_to_all:
        - gather_scatter:
    """

    def __init__(self, num_embeddings, embedding_dim, fused_op='all_to_all', *args, **kwargs):
        fused_op = fused_op.lower()
        assert fused_op in ['all_to_all', 'gather_scatter']

        super(FusedHybridParallelEmbeddingBag, self).__init__(num_embeddings, embedding_dim, *args, **kwargs)

        self.fused_op = fused_op

    def forward(self, input_, offsets=None, per_sample_weights=None, send_shape=None, scatter_dim=0, gather_dim=-1):
        output_parallel = F.embedding_bag(input_, self.weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)

        if self.output_device_type == 'cuda' and output_parallel.device.type == 'cpu':
            output_parallel = output_parallel.cuda()

        if send_shape is not None:
            output_parallel = output_parallel.view(*send_shape)

        if self.fused_op == 'all_to_all':
            # TODO: check situations when the scatter dim is indivisible by world size
            outputs = dual_all_to_all(output_parallel,
                                      self.parallel_mode,
                                      scatter_dim=scatter_dim,
                                      gather_dim=gather_dim)
        else:
            outputs = self.comm_func(output_parallel, self.parallel_mode, dim=gather_dim)
            outputs = split_forward_gather_backward(outputs, self.parallel_mode, dim=scatter_dim)

        return outputs


class ParallelQREmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_buckets: int,
    ):                 
        super().__init__()
        self.num_buckets = num_buckets
        self.q_embeddings = ColumnParallelEmbeddingBag(
            num_buckets,
            embedding_dim,
        )
        self.r_embeddings = ColumnParallelEmbeddingBag(
            num_buckets,
            embedding_dim,
        )

    def forward(self, x, offsets=None):
        if offsets is not None:
            x = x + x.new_tensor(offsets).unsqueeze(0)

        # Get the quotient index.
        quotient_index = torch.div(x, self.num_buckets, rounding_mode='floor')

        # Get the reminder index.
        remainder_index = torch.remainder(x, self.num_buckets)

        # Lookup the quotient_embedding using the quotient_index.
        quotient_embedding = self.q_embeddings(quotient_index)

        # Lookup the remainder_embedding using the remainder_index.
        remainder_embedding = self.r_embeddings(remainder_index)

        # Use multiplication as a combiner operation
        return quotient_embedding * remainder_embedding

    @property
    def weight(self):
        return self.q_embeddings.weight + self.r_embeddings.weight

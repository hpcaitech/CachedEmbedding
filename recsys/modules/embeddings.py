# This script is mainly inspired by torch.nn.Embedding & https://github.com/NVIDIA/Megatron-LM
import torch
import torch.nn.functional as F

from .. import distributed_manager as dist_manager
from .. import ParallelMode, distributed_logger as logger
from .functional import reduce_forward, tensor_gather_forward_split_backward, gather_forward_split_backward


def get_vocab_range(num_embeddings, rank, world_size):
    if world_size == 1:
        return 0, num_embeddings

    assert num_embeddings % world_size == 0
    chunk_size = num_embeddings // world_size
    return rank * chunk_size, (rank + 1) * chunk_size


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
                 _weight=None, device=None, dtype=None, parallel_mode=None,
                 init_method=torch.nn.init.xavier_normal_):  # I suppose this init method aligns with tensorflow?
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
        self.rank = dist_manager.get_rank(self.parallel_mode)
        self.world_size = dist_manager.get_world_size(self.parallel_mode)

        self.vocab_start_index, self.vocab_end_index = get_vocab_range(num_embeddings,
                                                                       self.rank,
                                                                       self.world_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # padding works
        if padding_idx is not None and self.vocab_start_index <= padding_idx < self.vocab_end_index:
            self.actual_padding_idx = padding_idx - self.vocab_start_index
        else:
            self.actual_padding_idx = None

        if _weight is None:
            self.weight = torch.nn.Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim, device=device, dtype=dtype
            ))

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
        output_parallel = F.embedding(masked_input,
                                      self.weight,
                                      self.actual_padding_idx,
                                      self.max_norm,
                                      self.norm_type,
                                      self.scale_grad_by_freq,
                                      self.sparse)

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

    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
                 mode='mean', include_last_offset=False,
                 _weight=None, device=None, dtype=None, parallel_mode=None,
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
        self.rank = dist_manager.get_rank(self.parallel_mode)
        self.world_size = dist_manager.get_world_size(self.parallel_mode)

        self.chunk_start_index, self.chunk_end_index, divisible = self.get_partition(embedding_dim,
                                                                                     self.rank,
                                                                                     self.world_size)
        self.embedding_dim_per_partition = self.chunk_end_index - self.chunk_start_index
        self.comm_func = gather_forward_split_backward if divisible else tensor_gather_forward_split_backward

        # Init weight
        if _weight is None:
            self.weight = torch.nn.Parameter(torch.empty(
                self.num_embeddings, self.embedding_dim_per_partition, device=device, dtype=dtype
            ))

            # TODO: check RNG states
            init_method(self.weight)
            if self.padding_idx is not None:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            chunk = torch.tensor_split(_weight, self.world_size, 1)[self.rank]
            assert list(chunk.shape) == [num_embeddings, self.embedding_dim_per_partition]
            self.weight = torch.nn.Parameter(chunk)

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
            return rank * chunk_size, (rank+1) * chunk_size, True

        logger.warning(
            f"Embedding dimension {embedding_dim} is not divisible by world size {world_size}. "
            f"torch.distributed.all_gather_object introducing additional copy operations is enabled",
            ranks=[0]
        )

        # align with the split strategy of torch.tensor_split
        size_list = [chunk_size + 1 if i < threshold else chunk_size for i in range(world_size)]
        offset = sum(size_list[:rank])
        return offset, offset+size_list[rank], False

    def forward(self, input_, offsets=None, per_sample_weights=None):
        output_parallel = F.embedding_bag(input_, self.weight, offsets,
                                          self.max_norm,
                                          self.norm_type,
                                          self.scale_grad_by_freq,
                                          self.mode,
                                          self.sparse,
                                          per_sample_weights,
                                          self.include_last_offset,
                                          self.padding_idx)
        output = self.comm_func(output_parallel, self.parallel_mode, dim=1)
        return output

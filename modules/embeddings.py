# This script is mainly inspired by torch.nn.Embedding & https://github.com/NVIDIA/Megatron-LM
import torch
import torch.nn.functional as F

from utils import get_world_size, get_rank, get_group, get_cpu_group


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
                 _weight=None, device=None, dtype=None,
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

        self.rank = get_rank()
        self.tp_world_size = get_world_size()

        self.vocab_start_index, self.vocab_end_index = get_vocab_range(num_embeddings,
                                                                       self.rank,
                                                                       self.tp_world_size)
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
        if self.tp_world_size > 1:
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
        if self.tp_world_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_forward(output_parallel)
        return output


class _ReduceForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if get_world_size() == 1:
            return x

        process_group = get_cpu_group() if x.device.type == 'cpu' else get_group()
        torch.distributed.all_reduce(x, group=process_group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


def reduce_forward(x):
    return _ReduceForward.apply(x)


class VocabParallelEmbeddingBag(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim,
                 padding_idx=None, max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False,
                 mode='mean', include_last_offset=False,
                 _weight=None, device=None, dtype=None,
                 init_method=torch.nn.init.xavier_normal_):
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

        self.mode = mode
        self.include_last_offset = include_last_offset

        self.rank = get_rank()
        self.tp_world_size = get_world_size()

        self.vocab_start_index, self.vocab_end_index = get_vocab_range(num_embeddings,
                                                                       self.rank,
                                                                       self.tp_world_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        # padding works fine
        if padding_idx is not None and self.vocab_start_index <= padding_idx < self.vocab_end_index:
            self.actual_padding_idx = padding_idx - self.vocab_start_index
        else:
            self.actual_padding_idx = None

        # Init weight
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

    def forward(self, input_, offsets=None, per_sample_weights=None):
        if self.tp_world_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            # masked_input = input_.clone() - self.vocab_start_index  # removing clone passes the test
            masked_input = input_ - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        output_parallel = F.embedding_bag(masked_input, self.weight, offsets,
                                          self.max_norm,
                                          self.norm_type,
                                          self.scale_grad_by_freq,
                                          self.mode,
                                          self.sparse,
                                          per_sample_weights,
                                          self.include_last_offset,
                                          self.actual_padding_idx)

        if self.tp_world_size > 1:
            output_parallel[input_mask, :] = 0.

        output = reduce_forward(output_parallel)
        return output

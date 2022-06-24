import numpy as np

import torch
import torch.nn as nn
from torch.profiler import record_function
from contextlib import nullcontext
import colossalai
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import distspec

from baselines.models.dlrm import DenseArch, OverArch, InteractionArch, choose
from ..utils import get_time_elapsed


def reshape_sparse_features(values):
    # this might introduce additional overhead for large batch
    return values.values().view(len(values.keys()), -1).transpose(0, 1)


class SparseArch(nn.Module):

    def __init__(self, num_embeddings_per_feature, embedding_dim, sparse=True, output_device_type=None):
        super(SparseArch, self).__init__()
        self.embed = nn.Embedding(sum(num_embeddings_per_feature), embedding_dim, sparse=sparse)

        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        self.register_buffer('offsets', torch.from_numpy(offsets).unsqueeze(0).requires_grad_(False), False)

        self.output_device_type = output_device_type

    def forward(self, sparse_features):
        x = reshape_sparse_features(sparse_features)
        embedding = self.embed(x + self.offsets)

        if self.output_device_type == "cuda" and self.offsets.device.type == "cpu":
            embedding = embedding.cuda()

        return embedding


class DLRM(nn.Module):

    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim,
        num_sparse_features,
        dense_in_features,
        dense_arch_layer_sizes,
        over_arch_layer_sizes,
    ):
        super(DLRM, self).__init__()

        self.sparse_arch = SparseArch(num_embeddings_per_feature, embedding_dim)
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
        )

    def forward(self, dense_features, sparse_features):
        logger = colossalai.logging.get_dist_logger()
        ctx1 = get_time_elapsed(logger, "embedding lookup in forward pass") \
            if gpc.config.inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                embedded_sparse = self.sparse_arch(sparse_features)

        ctx2 = get_time_elapsed(logger, "dense operations in forward pass") \
            if gpc.config.inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense MLP:"):
                embedded_dense = self.dense_arch(dense_features)

            with record_function("Feature interaction:"):
                concat_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
            with record_function("Output MLP:"):
                logits = self.over_arch(concat_dense)
        return logits


class FusedEmbeddingCollection(SparseArch):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 parallel_mode=ParallelMode.PARALLEL_1D,
                 *args,
                 **kwargs):
        super(FusedEmbeddingCollection, self).__init__(num_embeddings_per_feature, embedding_dim, *args, **kwargs)

        self.parallel_mode = parallel_mode
        self.world_size = gpc.get_world_size(parallel_mode)
        self.process_group = gpc.get_group(parallel_mode)

    def forward(self, sparse_features):
        x = reshape_sparse_features(sparse_features)
        output_ = self.embed(x + self.offsets)
        if self.output_device_type == "cuda" and self.offsets.device.type == "cpu":
            output_ = output_.cuda()
            output_.tensor_spec.dist_spec.process_group = self.process_group
        # print(f"Before all to all, shape: {output_.shape}, spec: {output_.tensor_spec.dist_spec}")
        output = output_.convert_to_dist_spec(distspec.shard(self.process_group, [0], [self.world_size]))
        # print(f"after all to all, shape: {output.shape}, spec: {output.tensor_spec.dist_spec}")
        return output


class FusedSparseModules(nn.Module):

    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim,
        reduction_mode='sum',
        parallel_mode=ParallelMode.PARALLEL_1D,
        sparse=True,
        output_device_type=None,
    ):
        super(FusedSparseModules, self).__init__()
        self.embed = nn.EmbeddingBag(sum(num_embeddings_per_feature),
                                     embedding_dim,
                                     sparse=sparse,
                                     mode=reduction_mode,
                                     include_last_offset=True)

        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        self.register_buffer('offsets', torch.from_numpy(offsets).requires_grad_(False), False)

        self.output_device = output_device_type
        self.group = gpc.get_group(parallel_mode)
        self.world_size = gpc.get_world_size(parallel_mode)

    def forward(self, sparse_features):
        keys = sparse_features.keys()
        assert len(keys) == len(self.offsets), f"keys len: {len(keys)}, offsets len: {len(self.offsets)}"

        sparse_dict = sparse_features.to_dict()
        flattened_sparse_features = torch.cat(
            [sparse_dict[key].values() + offset for key, offset in zip(keys, self.offsets)])
        batch_offsets = sparse_features.offsets()

        batch_size = len(sparse_features.lengths()) // len(keys)
        flattened_sparse_features = self.embed(flattened_sparse_features, batch_offsets)

        if self.output_device == 'cuda' and self.offsets.device.type == 'cpu':
            flattened_sparse_features = flattened_sparse_features.cuda()
            flattened_sparse_features.spec.dist_spec.process_group = self.group
        # [batch size * feature size, hidden size / world size]
        # it must convert to [batch size, feature size, hidden size / world size] before all to all
        flattened_sparse_features = flattened_sparse_features.view(batch_size, len(keys), -1)
        # print(f"before shape: {flattened_sparse_features.shape}, spec: {flattened_sparse_features.spec.dist_spec}")
        flattened_sparse_features = flattened_sparse_features.convert_to_dist_spec(
            distspec.shard(self.group, [0], [self.world_size]))
        # print(f"after shape: {flattened_sparse_features.shape}, spec: {flattened_sparse_features.spec.dist_spec}")
        return flattened_sparse_features


class FusedDenseModules(nn.Module):

    def __init__(self, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes,
                 over_arch_layer_sizes):
        super(FusedDenseModules, self).__init__()

        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes)

    def forward(self, dense_features, embedded_sparse):
        embedded_dense = self.dense_arch(dense_features)
        concat_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
        logits = self.over_arch(concat_dense)
        return logits


class HybridParallelDLRM(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 num_sparse_features,
                 dense_in_features,
                 dense_arch_layer_sizes,
                 over_arch_layer_sizes,
                 parallel_mode=ParallelMode.PARALLEL_1D,
                 sparse=True,
                 output_device_type=None):
        super(HybridParallelDLRM, self).__init__()

        self.sparse_modules = FusedEmbeddingCollection(num_embeddings_per_feature,
                                                       embedding_dim,
                                                       parallel_mode=parallel_mode,
                                                       sparse=sparse,
                                                       output_device_type=output_device_type)
        self.dense_modules = FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features,
                                               dense_arch_layer_sizes, over_arch_layer_sizes)

    def forward(self, dense_features, sparse_features):
        logger = colossalai.logging.get_dist_logger()
        ctx1 = get_time_elapsed(logger, "embedding lookup in forward pass") \
            if gpc.config.inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                embedded_sparse = self.sparse_modules(sparse_features)

        ctx2 = get_time_elapsed(logger, "dense operations in forward pass") \
            if gpc.config.inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense operations:"):
                logits = self.dense_modules(dense_features, embedded_sparse)

        return logits

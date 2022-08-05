#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The infrastructures of DLRM are mainly inspired by TorchRec:
# https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py

from typing import List, Optional
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function

from baselines.models.dlrm import DenseArch, OverArch, InteractionArch, choose
from ..modules.embeddings import ColumnParallelEmbeddingBag, FusedHybridParallelEmbeddingBag, \
    ParallelCachedEmbeddingBag, ParallelFreqAwareEmbeddingBag
from .. import ParallelMode, DISTLogger, DISTMGR as dist_manager
from ..utils import get_time_elapsed
from ..datasets.utils import KJTAllToAll


class SparseArch(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 reduction_mode='sum',
                 parallel_mode=ParallelMode.DEFAULT,
                 sparse=False,
                 output_device_type=None):
        super(SparseArch, self).__init__()

        self.embed = ColumnParallelEmbeddingBag(sum(num_embeddings_per_feature),
                                                embedding_dim,
                                                sparse=sparse,
                                                mode=reduction_mode,
                                                parallel_mode=parallel_mode,
                                                include_last_offset=True,
                                                output_device_type=output_device_type)

        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        self.register_buffer('offsets', torch.from_numpy(offsets).requires_grad_(False), False)

    def forward(self, sparse_features):
        keys = sparse_features.keys()
        assert len(keys) == len(self.offsets), f"keys len: {len(keys)}, offsets len: {len(self.offsets)}"

        sparse_dict = sparse_features.to_dict()
        flattened_sparse_features = torch.cat(
            [sparse_dict[key].values() + offset for key, offset in zip(keys, self.offsets)])
        batch_offsets = sparse_features.offsets()

        batch_size = len(sparse_features.lengths()) // len(keys)
        flattened_sparse_embeddings = self.embed(flattened_sparse_features, batch_offsets)

        if self.offsets.device.type == 'cpu':
            flattened_sparse_embeddings = flattened_sparse_embeddings.cuda()
        return flattened_sparse_embeddings.view(batch_size, len(keys), -1)


class DLRM(nn.Module):

    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim,
        num_sparse_features,
        dense_in_features,
        dense_arch_layer_sizes,
        over_arch_layer_sizes,
        dense_device,
        sparse_device,
        parallel_mode=ParallelMode.DEFAULT,
        sparse=False,
    ):
        super(DLRM, self).__init__()
        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_arch = SparseArch(num_embeddings_per_feature,
                                      embedding_dim,
                                      parallel_mode=parallel_mode,
                                      sparse=sparse,
                                      output_device_type=dense_device.type).to(sparse_device)
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes).to(dense_device)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features).to(dense_device)
        over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
        ).to(dense_device)

    def forward(self, dense_features, sparse_features, inspect_time=False):
        ctx1 = get_time_elapsed(DISTLogger, "embedding lookup in forward pass") \
            if inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                embedded_sparse = self.sparse_arch(sparse_features)

        ctx2 = get_time_elapsed(DISTLogger, "dense operations in forward pass") \
            if inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense MLP:"):
                embedded_dense = self.dense_arch(dense_features)

            with record_function("Feature interaction:"):
                concat_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
            with record_function("Output MLP:"):
                logits = self.over_arch(concat_dense)
        return logits


def sparse_embedding_shape_hook(embeddings, feature_size, batch_size):
    return embeddings.view(feature_size, batch_size, -1).transpose(0, 1)


class FusedSparseModules(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 fused_op='all_to_all',
                 reduction_mode='sum',
                 parallel_mode=ParallelMode.DEFAULT,
                 sparse=False,
                 output_device_type=None,
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True):
        super(FusedSparseModules, self).__init__()
        if use_cache:
            # self.embed = ParallelCachedEmbeddingBag(sum(num_embeddings_per_feature),
            #                                         embedding_dim,
            #                                         cache_sets=cache_sets,
            #                                         cache_lines=cache_lines,
            #                                         sparse=sparse,
            #                                         mode=reduction_mode,
            #                                         include_last_offset=True,
            #                                         parallel_mode=parallel_mode)
            self.embed = ParallelFreqAwareEmbeddingBag(sum(num_embeddings_per_feature),
                                                       embedding_dim,
                                                       sparse=True,
                                                       mode=reduction_mode,
                                                       include_last_offset=True,
                                                       parallel_mode=parallel_mode)
            self.embed.preprocess(cache_lines, cache_sets, id_freq_map, warmup_ratio, buffer_size=buffer_size)
        else:
            self.embed = FusedHybridParallelEmbeddingBag(sum(num_embeddings_per_feature),
                                                         embedding_dim,
                                                         fused_op=fused_op,
                                                         mode=reduction_mode,
                                                         parallel_mode=parallel_mode,
                                                         sparse=sparse,
                                                         include_last_offset=True,
                                                         output_device_type=output_device_type)
        self.world_size = dist_manager.get_world_size(parallel_mode)
        if is_dist_dataloader:
            self.kjt_collector = KJTAllToAll(dist_manager.get_group(parallel_mode))
        else:
            self.kjt_collector = None

    def forward(self, sparse_features):
        if self.kjt_collector:
            with record_function("(zhg)KJT AllToAll collective"):
                sparse_features = self.kjt_collector.all_to_all(sparse_features)

        keys, batch_size = sparse_features.keys(), sparse_features.stride()

        flattened_sparse_embeddings = self.embed(
            sparse_features.values(),
            sparse_features.offsets(),
            shape_hook=lambda x: sparse_embedding_shape_hook(x, len(keys), batch_size))
        return flattened_sparse_embeddings


class FusedDenseModules(nn.Module):
    """
    Fusing dense operations of DLRM into a single module
    """

    def __init__(self, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes,
                 over_arch_layer_sizes):
        super(FusedDenseModules, self).__init__()
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes)

    def forward(self, dense_features, embedded_sparse_features):
        embedded_dense_features = self.dense_arch(dense_features)
        concat_dense = self.inter_arch(dense_features=embedded_dense_features, sparse_features=embedded_sparse_features)
        logits = self.over_arch(concat_dense)

        return logits


class HybridParallelDLRM(nn.Module):
    """
    Model parallelized Embedding followed by Data parallelized dense modules
    """

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 num_sparse_features,
                 dense_in_features,
                 dense_arch_layer_sizes,
                 over_arch_layer_sizes,
                 dense_device,
                 sparse_device,
                 parallel_mode=ParallelMode.DEFAULT,
                 sparse=False,
                 fused_op='all_to_all',
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True):

        super(HybridParallelDLRM, self).__init__()
        if use_cache and sparse_device.type != dense_device.type:
            raise ValueError(f"Sparse device must be the same as dense device, "
                             f"however we got {sparse_device.type} for sparse, {dense_device.type} for dense")

        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature,
                                                 embedding_dim,
                                                 fused_op=fused_op,
                                                 parallel_mode=parallel_mode,
                                                 sparse=sparse,
                                                 output_device_type=dense_device.type,
                                                 use_cache=use_cache,
                                                 cache_sets=cache_sets,
                                                 cache_lines=cache_lines,
                                                 id_freq_map=id_freq_map,
                                                 warmup_ratio=warmup_ratio,
                                                 buffer_size=buffer_size,
                                                 is_dist_dataloader=is_dist_dataloader).to(sparse_device)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features,
                                                          dense_arch_layer_sizes,
                                                          over_arch_layer_sizes).to(dense_device),
                                 device_ids=[dist_manager.get_rank(parallel_mode)],
                                 process_group=dist_manager.get_group(parallel_mode),
                                 gradient_as_bucket_view=True,
                                 broadcast_buffers=False,
                                 static_graph=True)

        # precompute for parallelized embedding
        param_amount = sum(num_embeddings_per_feature) * embedding_dim
        param_storage = self.sparse_modules.embed.weight.element_size() * param_amount
        param_amount += sum(p.numel() for p in self.dense_modules.parameters())
        param_storage += sum(p.numel() * p.element_size() for p in self.dense_modules.parameters())

        buffer_amount = sum(b.numel() for b in self.sparse_modules.buffers()) + \
                        sum(b.numel() for b in self.dense_modules.buffers())
        buffer_storage = sum(b.numel() * b.element_size() for b in self.sparse_modules.buffers()) + \
                         sum(b.numel() * b.element_size() for b in self.dense_modules.buffers())
        stat_str = f"Number of model parameters: {param_amount:,}, storage overhead: {param_storage/1024**3:.2f} GB. " \
                   f"Number of model buffers: {buffer_amount:,}, storage overhead: {buffer_storage/1024**3:.2f} GB."
        self.stat_str = stat_str

    def forward(self, dense_features, sparse_features, inspect_time=False):
        ctx1 = get_time_elapsed(DISTLogger, "embedding lookup in forward pass") \
            if inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                # B // world size, sparse feature dim, embedding dim
                embedded_sparse = self.sparse_modules(sparse_features)

        ctx2 = get_time_elapsed(DISTLogger, "dense operations in forward pass") \
            if inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense operations:"):
                # B // world size, 1
                logits = self.dense_modules(dense_features, embedded_sparse)

        return logits

    def model_stats(self, prefix=""):
        return f"{prefix}: {self.stat_str}"

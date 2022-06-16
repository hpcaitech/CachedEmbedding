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

from torchrec.modules.mlp import MLP

from ..modules.embeddings import ColumnParallelEmbeddingBag, FusedHybridParallelEmbeddingBag
from .. import ParallelMode, DISTLogger, DISTMGR as dist_manager
from ..utils import get_time_elapsed


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


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


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(in_features, layer_sizes, bias=True, activation="relu", device=device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.register_buffer('triu_indices',
                             torch.triu_indices(self.F + 1, self.F + 1, offset=1).requires_grad_(False), False)

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        # b f+1 d
        combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(combined_values, torch.transpose(combined_values, 1, 2))
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


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


class FusedSparseModules(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 fused_op='all_to_all',
                 reduction_mode='sum',
                 parallel_mode=ParallelMode.DEFAULT,
                 sparse=False,
                 output_device_type=None):
        super(FusedSparseModules, self).__init__()
        self.embed = FusedHybridParallelEmbeddingBag(sum(num_embeddings_per_feature),
                                                     embedding_dim,
                                                     fused_op=fused_op,
                                                     reduction_mode=reduction_mode,
                                                     parallel_mode=parallel_mode,
                                                     sparse=sparse,
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
        return flattened_sparse_embeddings.view(batch_size, len(keys), -1)


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
                 fused_op='all_to_all'):

        super(HybridParallelDLRM, self).__init__()
        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature,
                                                 embedding_dim,
                                                 fused_op,
                                                 parallel_mode=parallel_mode,
                                                 sparse=sparse,
                                                 output_device_type=dense_device.type).to(sparse_device)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features,
                                                          dense_arch_layer_sizes,
                                                          over_arch_layer_sizes).to(dense_device),
                                 device_ids=[dist_manager.get_rank(parallel_mode)],
                                 process_group=dist_manager.get_group(parallel_mode),
                                 gradient_as_bucket_view=True,
                                 broadcast_buffers=False,
                                 static_graph=True)

    def forward(self, dense_features, sparse_features, inspect_time=False):
        """
        dense_features:     B // world size, dense feature dim
        sparse_features:    B,               sparse feature dim
        """
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

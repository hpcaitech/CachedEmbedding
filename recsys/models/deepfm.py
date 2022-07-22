#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.modules.deepfm import DeepFM, FactorizationMachine
from torchrec.sparse.jagged_tensor import KeyedTensor
from torch.fx import wrap

from recsys.modules.embeddings import ParallelMixVocabEmbeddingBag

# pyre-ignore[56]: Pyre was not able to infer the type of the decorator `torch.fx.wrap`.
@wrap
def _get_flatten_input(inputs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [input.flatten(1) for input in inputs],
        dim=1,
    )

class DeepFM(nn.Module):
    def __init__(
        self,
        dense_module: nn.Module,
    ) -> None:
        super().__init__()
        self.dense_module = dense_module

    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        deepfm_input = _get_flatten_input(embeddings)
        deepfm_output = self.dense_module(deepfm_input)
        return deepfm_output


class FactorizationMachine(nn.Module):
    r"""
    This is the Factorization Machine module, mentioned in the `DeepFM paper
    <https://arxiv.org/pdf/1703.04247.pdf>`_:

    This module does not cover the end-end functionality of the published paper.
    Instead, it covers only the FM part of the publication, and is used to learn
    2nd-order feature interactions.

    To support modeling flexibility, we customize the key components as:

        * Different from the public paper, we change the input from raw sparse
            features to embeddings of the features. It allows flexibility in embedding
            dimensions and the number of embeddings, as long as all embedding tensors
            have the same batch size.

    The general architecture of the module is like::

        # 1 x 1 output
        # ^
        # pass into `dense_module`
        # ^
        # 1 x 90
        # ^
        # concat
        # ^
        # 1 x 20, 1 x 30, 1 x 40 list of embeddings

    Example::

        batch_size = 3
        # the input embedding are in torch.Tensor of [batch_size, num_embeddings, embedding_dim]
        input_embeddings = [
            torch.randn(batch_size, 2, 64),
            torch.randn(batch_size, 2, 32),
        ]
        fm = FactorizationMachine()
        output = fm(embeddings=input_embeddings)
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            embeddings: List[torch.Tensor]:
                The list of all embeddings (e.g. dense, common_sparse,
                specialized_sparse, embedding_features, raw_embedding_features) in the
                shape of::

                    (batch_size, num_embeddings, embedding_dim)

                For the ease of operation, embeddings that have the same embedding
                dimension have the option to be stacked into a single tensor. For
                example, when we have 1 trained embedding with dimension=32, 5 native
                embeddings with dimension=64, and 3 dense features with dimension=16, we
                can prepare the embeddings list to be the list of::

                    tensor(B, 1, 32) (trained_embedding with num_embeddings=1, embedding_dim=32)
                    tensor(B, 5, 64) (native_embedding with num_embeddings=5, embedding_dim=64)
                    tensor(B, 3, 16) (dense_features with num_embeddings=3, embedding_dim=32)

                NOTE:
                    batch_size of all input tensors need to be identical.

        Returns:
            torch.Tensor: output of fm with flattened and concatenated `embeddings` as input. Expected to be [B, 1].
        """

        # flatten each embedding to be [B, N, D] -> [B, N*D], then cat them all on dim=1
        fm_input = _get_flatten_input(embeddings)
        sum_of_input = torch.sum(fm_input, dim=1, keepdim=True)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        square_of_sum = sum_of_input * sum_of_input
        cross_term = square_of_sum - sum_of_square
        cross_term = torch.sum(cross_term, dim=1, keepdim=True) * 0.5  # [B, 1]
        return cross_term


class SparseArch(nn.Module):
    """
    Processes the sparse features of the DeepFMNN model. Does embedding lookups for all
    EmbeddingBag and embedding features of each collection.
    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.
    Example::
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor):
        Returns:
            KeyedJaggedTensor: an output KJT of size F * D X B.
        """
        return self.embedding_bag_collection(features)


class DenseArch(nn.Module):
    """
    Processes the dense features of DeepFMNN model. Output layer is sized to
    the embedding_dimension of the EmbeddingBagCollection embeddings.
    Args:
        in_features (int): dimensionality of the dense input features.
        hidden_layer_size (int): sizes of the hidden layers in the DenseArch.
        embedding_dim (int): the same size of the embedding_dimension of sparseArch.
        device (torch.device): default compute device.
    Example::
        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=10, hidden_layer_size=10, embedding_dim=D)
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        hidden_layer_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): size B X `num_features`.
        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class FMInteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features) and apply the general DeepFM interaction according to the
    external source of DeepFM paper: https://arxiv.org/pdf/1703.04247.pdf
    The output dimension is expected to be a cat of `dense_features`, D.
    Args:
        fm_in_features (int): the input dimension of `dense_module` in DeepFM. For
            example, if the input embeddings is [randn(3, 2, 3), randn(3, 4, 5)], then
            the `fm_in_features` should be: 2 * 3 + 4 * 5.
        sparse_feature_names (List[str]): length of F.
        deep_fm_dimension (int): output of the deep interaction (DI) in the DeepFM arch.
    Example::
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        fm_inter_arch = FMInteractionArch(sparse_feature_names=keys)
        dense_features = torch.rand((B, D))
        sparse_features = KeyedTensor(
            keys=keys,
            length_per_key=[D, D],
            values=torch.rand((B, D * F)),
        )
        cat_fm_output = fm_inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        fm_in_features: int,
        sparse_feature_names: List[str],
        deep_fm_dimension: int,
    ) -> None:
        super().__init__()
        self.sparse_feature_names: List[str] = sparse_feature_names
        self.deep_fm = DeepFM(
            dense_module=nn.Sequential(
                nn.Linear(fm_in_features, deep_fm_dimension),
                nn.ReLU(),
            )
        )
        self.fm = FactorizationMachine()

    def forward(
        self, dense_features: torch.Tensor, sparse_features: KeyedTensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): tensor of size B X D.
            sparse_features (KeyedJaggedTensor): KJT of size F * D X B.
        Returns:
            torch.Tensor: an output tensor of size B X (D + DI + 1).
        """
        if len(self.sparse_feature_names) == 0:
            return dense_features

        tensor_list: List[torch.Tensor] = [dense_features]
        # dense/sparse interaction
        # size B X F
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])

        deep_interaction = self.deep_fm(tensor_list)
        fm_interaction = self.fm(tensor_list)

        return torch.cat([dense_features, deep_interaction, fm_interaction], dim=1)


class OverArch(nn.Module):
    """
    Final Arch - simple MLP. The output is just one target.
    Args:
        in_features (int): the output dimension of the interaction arch.
    Example::
        B = 20
        over_arch = OverArch()
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.model: nn.Module = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):
        Returns:
            torch.Tensor: an output tensor of size B X 1.
        """
        return self.model(features)

def sparse_embedding_shape_hook(embeddings, feature_size, batch_size):
    return embeddings.view(feature_size, batch_size, -1).transpose(0, 1)


class FusedSparseModules(nn.Module):

    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim,
        fused_op='all_to_all',
        reduction_mode='sum',
        parallel_mode=ParallelMode.DEFAULT,
        sparse=False,
        output_device_type=None,
    ):
        super(FusedSparseModules, self).__init__()
        self.embed = ParallelMixVocabEmbeddingBag(num_embeddings_per_feature,
                                                    embedding_dim,
                                                    fused_op=fused_op,
                                                    mode=reduction_mode,
                                                    parallel_mode=parallel_mode,
                                                    sparse=sparse,
                                                    include_last_offset=True,
                                                    output_device_type=output_device_type)
        self.world_size = dist_manager.get_world_size(parallel_mode)
        self.kjt_collector = KJTAllToAll(dist_manager.get_group(parallel_mode))

    def forward(self, sparse_features):
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
        fm_in_features = embedding_dim
        for conf in embedding_bag_collection.embedding_bag_configs():
            for feat in conf.feature_names:
                feature_names.append(feat)
                fm_in_features += conf.embedding_dim

        self.dense_arch = DenseArch(
            in_features=num_dense_features,
            hidden_layer_size=hidden_layer_size,
            embedding_dim=embedding_dim,
        )
        self.inter_arch = FMInteractionArch(
            fm_in_features=fm_in_features,
            sparse_feature_names=feature_names,
            deep_fm_dimension=deep_fm_dimension,
        )
        over_in_features = embedding_dim + deep_fm_dimension + 1
        self.over_arch = OverArch(over_in_features)
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)

    def forward(self, dense_features, embedded_sparse_features):
        embedded_dense_features = self.dense_arch(dense_features)
        concat_dense = self.inter_arch(dense_features=embedded_dense_features, sparse_features=embedded_sparse_features)
        logits = self.over_arch(concat_dense)

        return logits

class HybridParallelDeepFM(nn.Module):
    """
    Basic recsys module with DeepFM arch. Processes sparse features by
    learning pooled embeddings for each feature. Learns the relationship between
    dense features and sparse features by projecting dense features into the same
    embedding space. Learns the interaction among those dense and sparse features
    by deep_fm proposed in this paper: https://arxiv.org/pdf/1703.04247.pdf
    The module assumes all sparse features have the same embedding dimension
    (i.e, each `EmbeddingBagConfig` uses the same embedding_dim)
    The following notation is used throughout the documentation for the models:
    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features
    Args:
        num_dense_features (int): the number of input dense features.
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        hidden_layer_size (int): the hidden layer size used in dense module.
        deep_fm_dimension (int): the output layer size used in `deep_fm`'s deep
            interaction module.
    Example::
        B = 2
        D = 8
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = SimpleDeepFMNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )
        features = torch.rand((B, 100))
        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )
        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim,
        num_sparse_features,
        num_dense_features,
        embedding_bag_collection: EmbeddingBagCollection,
        hidden_layer_size: int,
        deep_fm_dimension: int,
        dense_device,
        sparse_device,
        parallel_mode=ParallelMode.DEFAULT,
        sparse=False,
        fused_op='all_to_all',
    ) -> None:
        super().__init__()
        
        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature,
                                                 embedding_dim,
                                                 fused_op=fused_op,
                                                 parallel_mode=parallel_mode,
                                                 sparse=sparse,
                                                 output_device_type=dense_device.type).to(sparse_device)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, num_dense_features,
                                                          hidden_layer_size,
                                                          deep_fm_dimension).to(dense_device),
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


# import math

# import torch
# import torch.nn as nn
# from torch.profiler import record_function

# from recsys import ParallelMode, DISTLogger, DISTMGR
# from recsys.modules.embeddings import ParallelMixVocabEmbeddingBag
# from colo_recsys.utils import count_parameters


# class FeatureEmbedding(nn.Module):
    
#     def __init__(self, field_dims, emb_dim, enable_qr):
#         super().__init__()
#         self.embedding = ParallelMixVocabEmbeddingBag(field_dims, emb_dim, mode='mean',
#                                                           parallel_mode=ParallelMode.TENSOR_PARALLEL,
#                                                           enable_qr=enable_qr, do_fair=True)
            
#         # print('Saved params (M)',emb_dim*(sum(field_dims) - math.ceil(math.sqrt(sum(field_dims))))//1_000_000)

#     def forward(self,sparse_features):
#         return self.embedding(sparse_features)
    

# class FeatureLinear(nn.Module):

#     def __init__(self, dense_input_dim, output_dim=1):
#         super().__init__()
#         self.linear = nn.Linear(dense_input_dim, output_dim)
#         self.bias = nn.Parameter(torch.zeros((output_dim,)))
        
#     def forward(self,x):
#         return self.linear(x) + self.bias


# class FactorizationMachine(nn.Module):

#     def __init__(self, reduce_sum=True):
#         super().__init__()
#         self.reduce_sum = reduce_sum
        
#     def forward(self, x):
#         square_of_sum = torch.sum(x, dim=1)**2
#         sum_of_square = torch.sum(x**2, dim=1)
#         ix = square_of_sum - sum_of_square

#         return 0.5 * ix


# class MultiLayerPerceptron(nn.Module):

#     def __init__(self, emb_dims, dropout, output_layer=True):
#         super().__init__()
#         layers = []

#         for i in range(len(emb_dims)-1):
#             layers.append(nn.Linear(emb_dims[i],emb_dims[i+1]))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout))

#         if output_layer:
#             layers.append(nn.Linear(emb_dims[-1],1))

#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         with record_function('MLP layer'):
#             return self.mlp(x)
        
        
# class DeepFactorizationMachine(nn.Module):
    
#     def __init__(self, num_embed_per_feature, dense_input_dim, embed_dim, mlp_dims, dropout, enable_qr):
#         super().__init__()
#         self.linear = FeatureLinear(dense_input_dim, embed_dim)
#         self.fm = FactorizationMachine(reduce_sum=True)
#         self.embedding = FeatureEmbedding(num_embed_per_feature, embed_dim, enable_qr)
#         self.mlp = MultiLayerPerceptron([embed_dim*2]+mlp_dims, dropout)
    
#     def forward(self, sparse_feats, dense_feats):
#         """
#         :param x: Long tensor of size ``(batch_size, num_fields)``
#         """
#         embed_x = self.embedding(sparse_feats)
#         linear_x = self.linear(dense_feats)
#         combined_x = torch.cat([embed_x, linear_x], dim=1)
#         x = self.fm(embed_x) + self.mlp(combined_x).squeeze(-1)
#         return torch.sigmoid(x)

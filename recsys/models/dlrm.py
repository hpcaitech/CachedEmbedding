# The infrastructures of DLRM are mainly inspired by TorchRec:
# https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py
import os
import torch
from contextlib import nullcontext
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import record_function
from typing import List
from baselines.models.dlrm import DenseArch, OverArch, InteractionArch, choose
from ..utils import get_time_elapsed
from ..datasets.utils import KJTAllToAll

import colossalai
from colossalai.nn.parallel.layers import ParallelFreqAwareEmbeddingBag, EvictionStrategy, \
    TablewiseEmbeddingBagConfig, ParallelFreqAwareEmbeddingBagTablewise
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
import numpy as np

dist_logger = colossalai.logging.get_dist_logger()


def sparse_embedding_shape_hook(embeddings, feature_size, batch_size):
    return embeddings.view(feature_size, batch_size, -1).transpose(0, 1)


def sparse_embedding_shape_hook_for_tablewise(embeddings, feature_size, batch_size):
    return embeddings.view(embeddings.shape[0], feature_size, -1)


def prepare_tablewise_config(num_embeddings_per_feature,
                             cache_ratio,
                             id_freq_map_total=None,
                             dataset="criteo_kaggle",
                             world_size=2):
    # WARNING, prototype. only support criteo_kaggle dataset and world_size == 2, 4
    # TODO: automatic arrange
    embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
    if 'criteo' in dataset and 'kaggle' in dataset:
        if world_size == 1:
            rank_arrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif world_size == 2:
            rank_arrange = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        elif world_size == 4:
            rank_arrange = [3, 1, 0, 3, 1, 0, 2, 1, 0, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 0, 2, 0, 0, 2, 3, 2]
        else :
            raise NotImplementedError("Other Tablewise settings are under development")

        table_offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)])

        for i, num_embeddings in enumerate(num_embeddings_per_feature):
            ids_freq_mapping = None
            if id_freq_map_total != None:
                ids_freq_mapping = id_freq_map_total[table_offsets[i] : table_offsets[i + 1]]
            cuda_row_num = int(cache_ratio * num_embeddings) + 2000
            if cuda_row_num > num_embeddings:
                cuda_row_num = num_embeddings
            embedding_bag_config_list.append(
                TablewiseEmbeddingBagConfig(
                    num_embeddings=num_embeddings,
                    cuda_row_num=cuda_row_num,
                    assigned_rank=rank_arrange[i],
                    ids_freq_mapping=ids_freq_mapping
                )
            )
        return embedding_bag_config_list
    else:
        raise NotImplementedError("Other Tablewise settings are under development")


class FusedSparseModules(nn.Module):

    def __init__(self,
                 num_embeddings_per_feature,
                 embedding_dim,
                 fused_op='all_to_all',
                 reduction_mode='sum',
                 sparse=False,
                 output_device_type=None,
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True,
                 use_lfu_eviction=False,
                 use_tablewise_parallel=False,
                 dataset: str = None):
        super(FusedSparseModules, self).__init__()

        if use_cache:
            if use_tablewise_parallel:
                # establist config list
                world_size = torch.distributed.get_world_size()
                embedding_bag_config_list = prepare_tablewise_config(
                    num_embeddings_per_feature, 0.3, id_freq_map, dataset, world_size)
                self.embed = ParallelFreqAwareEmbeddingBagTablewise(
                    embedding_bag_config_list,
                    embedding_dim,
                    sparse=sparse,
                    mode=reduction_mode,
                    include_last_offset=True,
                    evict_strategy=EvictionStrategy.LFU if use_lfu_eviction else EvictionStrategy.DATASET
                )
                self.shape_hook = sparse_embedding_shape_hook_for_tablewise
            else:
                self.embed = ParallelFreqAwareEmbeddingBag(
                    sum(num_embeddings_per_feature),
                    embedding_dim,
                    sparse=sparse,
                    mode=reduction_mode,
                    include_last_offset=True,
                    cuda_row_num=cache_sets,
                    ids_freq_mapping=id_freq_map,
                    warmup_ratio=warmup_ratio,
                    buffer_size=buffer_size,
                    evict_strategy=EvictionStrategy.LFU if use_lfu_eviction else EvictionStrategy.DATASET
                )
                self.shape_hook = sparse_embedding_shape_hook
        else:
            raise NotImplementedError("Other EmbeddingBags are under development")

        if is_dist_dataloader:
            self.kjt_collector = KJTAllToAll(gpc.get_group(ParallelMode.GLOBAL))
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
            shape_hook=lambda x: self.shape_hook(x, len(keys), batch_size))
        return flattened_sparse_embeddings


class FusedDenseModules(nn.Module):
    """
    Fusing dense operations of DLRM into a single module
    """

    def __init__(self, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes,
                 over_arch_layer_sizes):
        super(FusedDenseModules, self).__init__()
        if dense_in_features <= 0:
            self.dense_arch = nn.Identity()
            over_in_features = choose(num_sparse_features, 2)
            num_dense = 0
        else:
            self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
            over_in_features = (embedding_dim + choose(num_sparse_features, 2) + num_sparse_features)
            num_dense = 1

        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features, num_dense_features=num_dense)
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
                 sparse=False,
                 fused_op='all_to_all',
                 use_cache=False,
                 cache_sets=500_000,
                 cache_lines=1,
                 id_freq_map=None,
                 warmup_ratio=0.7,
                 buffer_size=50_000,
                 is_dist_dataloader=True,
                 use_lfu_eviction=False,
                 use_tablewise=False,
                 dataset: str = None):

        super(HybridParallelDLRM, self).__init__()
        if use_cache and sparse_device.type != dense_device.type:
            raise ValueError(f"Sparse device must be the same as dense device, "
                             f"however we got {sparse_device.type} for sparse, {dense_device.type} for dense")

        self.dense_device = dense_device
        self.sparse_device = sparse_device

        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature,
                                                 embedding_dim,
                                                 fused_op=fused_op,
                                                 sparse=sparse,
                                                 output_device_type=dense_device.type,
                                                 use_cache=use_cache,
                                                 cache_sets=cache_sets,
                                                 cache_lines=cache_lines,
                                                 id_freq_map=id_freq_map,
                                                 warmup_ratio=warmup_ratio,
                                                 buffer_size=buffer_size,
                                                 is_dist_dataloader=is_dist_dataloader,
                                                 use_lfu_eviction=use_lfu_eviction,
                                                 use_tablewise_parallel=use_tablewise,
                                                 dataset=dataset
                                                 ).to(sparse_device)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features,
                                                          dense_arch_layer_sizes,
                                                          over_arch_layer_sizes).to(dense_device),
                                 device_ids=[0 if os.environ.get("NVT_TAG", None) else gpc.get_global_rank()],
                                 process_group=gpc.get_group(ParallelMode.GLOBAL),
                                 gradient_as_bucket_view=True,
                                 broadcast_buffers=False,
                                 static_graph=True)

        # precompute for parallelized embedding
        param_amount = sum(num_embeddings_per_feature) * embedding_dim
        param_storage = self.sparse_modules.embed.element_size() * param_amount
        param_amount += sum(p.numel() for p in self.dense_modules.parameters())
        param_storage += sum(p.numel() * p.element_size() for p in self.dense_modules.parameters())
#
        buffer_amount = sum(b.numel() for b in self.sparse_modules.buffers()) + \
            sum(b.numel() for b in self.dense_modules.buffers())
        buffer_storage = sum(b.numel() * b.element_size() for b in self.sparse_modules.buffers()) + \
            sum(b.numel() * b.element_size() for b in self.dense_modules.buffers())
        stat_str = f"Number of model parameters: {param_amount:,}, storage overhead: {param_storage/1024**3:.2f} GB. " \
                   f"Number of model buffers: {buffer_amount:,}, storage overhead: {buffer_storage/1024**3:.2f} GB."
        self.stat_str = stat_str

    def forward(self, dense_features, sparse_features, inspect_time=False):
        ctx1 = get_time_elapsed(dist_logger, "embedding lookup in forward pass") \
            if inspect_time else nullcontext()
        with ctx1:
            with record_function("Embedding lookup:"):
                # B // world size, sparse feature dim, embedding dim
                embedded_sparse = self.sparse_modules(sparse_features)

        ctx2 = get_time_elapsed(dist_logger, "dense operations in forward pass") \
            if inspect_time else nullcontext()
        with ctx2:
            with record_function("Dense operations:"):
                # B // world size, 1
                logits = self.dense_modules(dense_features, embedded_sparse)

        return logits

    def model_stats(self, prefix=""):
        return f"{prefix}: {self.stat_str}"

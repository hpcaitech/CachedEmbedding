import torch
import torch.nn as nn
from torch.profiler import record_function
from contextlib import nullcontext
import colossalai
from colossalai.core import global_context as gpc

from modules.colossal_embedding import EmbeddingCollection
from models.dlrm import DenseArch, OverArch, InteractionArch, choose
from utils import get_time_elapsed


def reshape_spare_features(values):
    # this might introduce additional overhead for large batch
    return values.values().view(len(values.keys()), -1).transpose(0, 1)


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

        self.sparse_arch = EmbeddingCollection(num_embeddings_per_feature, embedding_dim)
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

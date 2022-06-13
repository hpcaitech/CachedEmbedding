import numpy as np
import torch
import torch.nn as nn

import colossalai
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils.cuda import get_current_device

from .colossal_functional import copy_to_gpu


class EmbeddingCollection(nn.Embedding):

    def __init__(self, num_embeddings_per_feature, embedding_dim):
        tot_features = sum(num_embeddings_per_feature)
        tp_size = gpc.tensor_parallel_size
        if tot_features % tp_size != 0:
            tot_features += (tp_size - tot_features % tp_size)
        super().__init__(tot_features, embedding_dim)

        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        self.register_buffer('offsets', torch.from_numpy(offsets).unsqueeze(0).requires_grad_(False), False)

        # if device is None:
        #     raise ValueError("Please explicitly set the device")
        # self._use_cpu = False if device.type == 'cuda' else True
        #
        # self.to(device)

    def forward(self, x):
        embedding = super().forward(x + self.offsets)
        if self.offsets.device.type == 'cpu':
            embedding = copy_to_gpu(embedding)
        return embedding


def main():
    config = dict()
    colossalai.launch_from_torch(config=config, verbose=False)

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()

    logger.info(f"launch rank {gpc.get_global_rank()}, Done, "
                f"DP size: {gpc.get_world_size(ParallelMode.DATA)}, "
                f"MP size: {gpc.get_world_size(ParallelMode.MODEL)}")

    device = get_current_device()

    torch.manual_seed(gpc.get_global_rank())
    f1 = torch.randint(6, (2, 1))
    f2 = torch.randint(4, (2, 1))
    idx = torch.cat([f1, f2], dim=1).to(device)
    # offsets = torch.tensor([0, *np.cumsum([6, 4])[:-1]], dtype=torch.int64).unsqueeze(0)
    logger.info(f"Rank: {gpc.get_global_rank()}, idx: {idx}")

    model = EmbeddingCollection([6, 4], 2, device)

    res = model(idx).view(2, -1)    # 2, 2 x 3
    logger.info(f"Rank: {gpc.get_global_rank()}, res: {res}")

    loss = torch.prod(res, dim=0).sum()
    loss.backward()
    logger.info(f"{model.weight.grad}")


if __name__ == "__main__":
    main()

import numpy as np
import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import colossalai
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from colossalai.tensor import ColoTensor, TensorSpec, ComputePattern, ParallelAction, DistSpecManager
from colossalai.tensor import distspec
from colossalai.utils import ColoInitContext
from colossalai.utils.cuda import get_current_device


class EmbeddingBag(nn.Module):
    def __init__(self, num_embeddings_per_feature, embedding_dim):
        super(EmbeddingBag, self).__init__()

        tot_features = sum(num_embeddings_per_feature)
        # TODO: Other mode
        tot_features += tot_features % gpc.get_world_size(ParallelMode.PARALLEL_1D)
        self.embed = nn.Embedding(tot_features, embedding_dim)
        offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])
        # TODO: check device
        self.offsets = torch.from_numpy(offsets).to(get_current_device()).unsqueeze(0)

    def forward(self, x):
        x = x + self.offsets
        return self.embed(x)


def main():
    config = dict(parallel=dict(tensor=dict(mode="1d", size=2),))
    colossalai.launch_from_torch(config=config, verbose=False)

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()

    mb = KeyedJaggedTensor(
        keys=["product", "user"],
        values=torch.tensor([4, 5, 0, 2]).cuda(),
        lengths=torch.tensor([1, 1, 1, 1], dtype=torch.int64).cuda(),
    )

    idx = mb.values().view(len(mb.keys()), -1).transpose(0, 1)
    logger.info(f"{idx}", ranks=[0])

    with ColoInitContext(device=get_current_device()):
        model = EmbeddingBag([6, 4], 3)
    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        model.embed.weight.set_spec(spec)

    res = model(idx).view(2, -1)  # 2, 2 x 3
    logger.info(f"Rank: {gpc.get_local_rank(ParallelMode.PARALLEL_1D)}, res: {res}")

    loss = torch.prod(res, dim=0).sum()
    loss.backward()
    logger.info(f"{model.weight.weight.grad}")


if __name__ == "__main__":
    main()

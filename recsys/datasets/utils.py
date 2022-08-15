import numpy as np
import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch


class KJTAllToAll:
    """
    Different from the module defined in torchrec.

    Basically, this class conducts all_gather with all_to_all collective.
    """

    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

    @torch.no_grad()
    def all_to_all(self, kjt):
        if self.world_size == 1:
            return kjt
        # TODO: add sample weights
        values, lengths = kjt.values(), kjt.lengths()
        keys, batch_size = kjt.keys(), kjt.stride()

        # collect global data
        length_list = [lengths if i == self.rank else lengths.clone() for i in range(self.world_size)]
        all_length_list = [torch.empty_like(lengths) for _ in range(self.world_size)]
        dist.all_to_all(all_length_list, length_list, group=self.group)

        intermediate_all_length_list = [_length.view(-1, batch_size) for _length in all_length_list]
        all_length_per_key_list = [torch.sum(_length, dim=1).cpu().tolist() for _length in intermediate_all_length_list]

        all_value_length = [torch.sum(each).item() for each in all_length_list]
        value_list = [values if i == self.rank else values.clone() for i in range(self.world_size)]
        all_value_list = [
            torch.empty(_length, dtype=values.dtype, device=values.device) for _length in all_value_length
        ]
        dist.all_to_all(all_value_list, value_list, group=self.group)

        all_value_list = [
            torch.split(_values, _length_per_key)    # [ key size, variable value size ]
            for _values, _length_per_key in zip(all_value_list, all_length_per_key_list)    # world size
        ]
        all_values = torch.cat([torch.cat(values_per_key) for values_per_key in zip(*all_value_list)])

        all_lengths = torch.cat(intermediate_all_length_list, dim=1).view(-1)
        return KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=all_values,
            lengths=all_lengths,
        )


class KJTTransform:

    def __init__(self, dataloader, hashes=None):
        self.batch_size = dataloader.batch_size
        self.cats = dataloader.cat_names
        self.conts = dataloader.cont_names
        self.labels = dataloader.label_names
        self.sparse_offset = torch.tensor(
            [0, *np.cumsum(hashes)[:-1]], dtype=torch.long, device=torch.cuda.current_device()).view(1, -1) \
            if hashes is not None else None

        _num_ids_in_batch = len(self.cats) * self.batch_size
        self.lengths = torch.ones((_num_ids_in_batch,), dtype=torch.int32)
        self.offsets = torch.arange(0, _num_ids_in_batch + 1, dtype=torch.int32)
        self.length_per_key = len(self.cats) * [self.batch_size]
        self.offset_per_key = [self.batch_size * i for i in range(len(self.cats) + 1)]
        self.index_per_key = {key: i for (i, key) in enumerate(self.cats)}

    def transform(self, batch):
        sparse, dense = [], []
        for col in self.cats:
            sparse.append(batch[0][col])
        sparse = torch.cat(sparse, dim=1)
        if self.sparse_offset is not None:
            sparse += self.sparse_offset
        for col in self.conts:
            dense.append(batch[0][col])
        dense = torch.cat(dense, dim=1)

        return Batch(
            dense_features=dense,
            sparse_features=KeyedJaggedTensor(
                keys=self.cats,
                values=sparse.transpose(1, 0).contiguous().view(-1),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.batch_size,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=batch[1],
        )

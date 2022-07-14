import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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

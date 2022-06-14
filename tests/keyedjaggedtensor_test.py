import numpy as np
import torch
import torchrec


def test_keyedjaggedtensor():
    device = torch.device('cuda')
    num_embeddings_per_feature = [1000, 987, 1874]
    feature_offsets = torch.from_numpy(np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])).to(device)

    # synthesize a batch of 2 samples:
    #
    #     feature 1    |    feature 2    | feature 3
    # [id1, id2, id3]  |   [id1, ]       | [id1, id2]
    # []               |   [id1, id2]    | [id1, ]

    lengths = [3, 0, 1, 2, 2, 1]
    local_inputs = []
    target = []
    batch_size = len(lengths) // len(num_embeddings_per_feature)
    for idx, l in enumerate(lengths):
        high = num_embeddings_per_feature[idx // batch_size]
        offset = feature_offsets[idx // batch_size]

        _local = torch.randint(low=0, high=high, size=(l,), dtype=torch.long, device=device)
        local_inputs.append(_local)
        target.append(_local + offset)

    inputs = torchrec.KeyedJaggedTensor(
        keys=["t_1", "t_2", "t_3"],
        values=torch.cat(local_inputs),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )
    print(inputs)
    print(f"Offsets: {inputs.offsets()}")
    keys = inputs.keys()
    assert len(keys) == len(feature_offsets)

    feat_dict = inputs.to_dict()
    flattened = torch.cat([feat_dict[key].values() + offset for key, offset in zip(keys, feature_offsets)])

    target = torch.cat(target)
    assert torch.allclose(flattened, target)


if __name__ == "__main__":
    test_keyedjaggedtensor()

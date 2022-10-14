import torch
import numpy as np
from colossalai.nn.parallel.layers.cache_embedding import CachedEmbeddingBag
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from fbgemm_gpu.split_table_batched_embeddings_ops import SplitTableBatchedEmbeddingBagsCodegen, EmbeddingLocation, ComputeDevice, CacheAlgorithm
import time

########### GLOBAL SETTINGS ##################


BATCH_SIZE = 65536
TABLLE_NUM = 856
FILE_LIST = [f"/data/scratch/RecSys/embedding_bag/fbgemm_t856_bs65536_{i}.pt" for i in range(16)]
KEYS = []
for i in range(TABLLE_NUM):
    KEYS.append("table_{}".format(i))
EMBEDDING_DIM = 128
# Full dataset is too big
# CHOSEN_TABLES = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 18, 22, 27, 28]
# CHOSEN_TABLES = [5, 8, 37, 54, 71,72,73,74,85,86,89,95,96,97,107,131,163, 185, 196, 204, 211, ]
CHOSEN_TABLES = [i for i in range(300,418)]
TEST_ITER = 100
TEST_BATCH_SIZE = 16384
WARMUP_ITERS = 5
##############################################


def load_file(file_path):
    indices, offsets, lengths = torch.load(file_path)
    indices = indices.int().cuda()
    offsets = offsets.int().cuda()
    lengths = lengths.int().cuda()
    num_embeddings_per_table = []
    indices_per_table = []
    lengths_per_table = []
    offsets_per_table = []
    for i in range(TABLLE_NUM):
        if i not in CHOSEN_TABLES:
            continue
        start_pos = offsets[i * BATCH_SIZE]
        end_pos = offsets[i * BATCH_SIZE + BATCH_SIZE]
        part = indices[start_pos:end_pos]
        indices_per_table.append(part)
        lengths_per_table.append(lengths[i])
        offsets_per_table.append(torch.cumsum(
            torch.cat((torch.tensor([0]).cuda(), lengths[i])), 0
        ))
        if part.numel() == 0:
            num_embeddings_per_table.append(0)
        else:
            num_embeddings_per_table.append(torch.max(part).int().item() + 1)
    return indices_per_table, offsets_per_table, lengths_per_table, num_embeddings_per_table


def load_file_kjt(file_path):
    indices, offsets, lengths = torch.load(file_path)
    length_per_key = []
    for i in range(TABLLE_NUM):
        length_per_key.append(lengths[i])
    ret = KeyedJaggedTensor(KEYS, indices, offsets=offsets,
                            length_per_key=length_per_key)
    return ret


def load_random_batch(indices_per_table, offsets_per_table, lengths_per_table, batch_size=4096):
    chosen_indices_list = []
    chosen_lengths_list = []
    choose = torch.randint(
        0, offsets_per_table[0].shape[0] - 1, (batch_size,)).cuda()
    for indices, offsets, lengths in zip(indices_per_table, offsets_per_table, lengths_per_table):

        chosen_lengths_list.append(lengths[choose])
        start_list = offsets[choose]
        end_list = offsets[choose + 1]
        chosen_indices_atoms = []
        for start, end in zip(start_list, end_list):
            chosen_indices_atoms.append(indices[start: end])
        chosen_indices_list.append(torch.cat(chosen_indices_atoms, 0))
    return chosen_indices_list, chosen_lengths_list


def merge_to_kjt(indices_list, lengths_list, length_per_key) -> KeyedJaggedTensor:
    values = torch.cat(indices_list)
    lengths = torch.cat(lengths_list)
    return KeyedJaggedTensor(
        keys=[KEYS[i] for i in CHOSEN_TABLES],
        values=values,
        lengths=lengths,
        length_per_key=length_per_key,
    )


def test(iter_num=1, batch_size=4096):
    print("loading file")
    indices_per_table, offsets_per_table, lengths_per_table, num_embeddings_per_table = load_file(
        FILE_LIST[0])
    table_idx_offset_list = np.cumsum([0] + num_embeddings_per_table)
    fae = CachedEmbeddingBag(
        num_embeddings=sum(num_embeddings_per_table),
        embedding_dim=EMBEDDING_DIM,
        sparse=True,
        include_last_offset=True,
        cache_ratio=0.05,
        pin_weight=True,
    )
    fae_forwarding_time = 0.0
    fae_backwarding_time = 0.0
    grad_fae = None
    managed_type = (EmbeddingLocation.MANAGED_CACHING)
    uvm = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[(
            num_embeddings,
            EMBEDDING_DIM,
            managed_type,
            ComputeDevice.CUDA,
        ) for num_embeddings in num_embeddings_per_table],
        cache_load_factor=0.05,
        cache_algorithm=CacheAlgorithm.LFU
    )
    uvm.init_embedding_weights_uniform(-0.5, 0.5)
    # print(sum(num_embeddings_per_table))
    # print(uvm.weights_uvm.shape)
    uvm_forwarding_time = 0.0
    uvm_backwarding_time = 0.0
    grad_uvm = None
    print("testing:")
    for iter in range(iter_num):
        # load batch
        chosen_indices_list, chosen_lengths_list = load_random_batch(indices_per_table,
                                                                     offsets_per_table, lengths_per_table, batch_size)
        features = merge_to_kjt(chosen_indices_list,
                                chosen_lengths_list, num_embeddings_per_table)
        print("iter {} batch loaded.".format(iter))

        # fae
        with torch.no_grad():
            values = features.values().long()
            offsets = features.offsets().long()
            weights = features.weights_or_none()
            batch_size = len(features.offsets()) // len(features.keys())
            if weights is not None and not torch.is_floating_point(weights):
                weights = None
            split_view = torch.tensor_split(
                values, features.offset_per_key()[1:-1], dim=0)
            for i, chunk in enumerate(split_view):
                torch.add(chunk, table_idx_offset_list[i], out=chunk)
        start = time.time()
        output = fae(values, offsets, weights)
        ret = torch.cat(output.split(batch_size), 1)
        if iter >= WARMUP_ITERS:
            fae_forwarding_time += time.time() - start
            print("fae forwarded. avg time = {} s".format(
                fae_forwarding_time / (iter + 1 - WARMUP_ITERS)))
        grad_fae = torch.randn_like(ret) if grad_fae is None else grad_fae
        start = time.time()
        ret.backward(grad_fae)
        if iter >= WARMUP_ITERS:
            fae_backwarding_time += time.time() - start
            print("fae backwarded. avg time = {} s".format(
                fae_backwarding_time / (iter + 1 - WARMUP_ITERS)))
        fae.zero_grad()
        
        # uvm 
        start = time.time()
        ret = uvm(features.values().long(), features.offsets().long())
        if iter >= WARMUP_ITERS:
            uvm_forwarding_time += time.time() - start
            print("uvm forwarded. avg time = {} s".format(
                uvm_forwarding_time / (iter + 1 - WARMUP_ITERS)))
        grad_uvm = torch.randn_like(ret) if grad_uvm is None else grad_uvm
        start = time.time()
        ret.backward(grad_uvm)
        if iter >= WARMUP_ITERS:
            uvm_backwarding_time += time.time() - start
            print("uvm backwarded. avg time = {} s".format(
                uvm_backwarding_time / (iter + 1 - WARMUP_ITERS)))
        uvm.zero_grad()


# test(TEST_ITER, TEST_BATCH_SIZE)
num_embeddings_per_table = None
for i in range(16):
    indices_per_table, offsets_per_table, lengths_per_table, num_embeddings_per_table1 = load_file(FILE_LIST[i])
    if num_embeddings_per_table == None:
        num_embeddings_per_table = num_embeddings_per_table1
    else:
        for i, num in enumerate(num_embeddings_per_table1):
            num_embeddings_per_table[i] = max(num_embeddings_per_table[i], num)
    print(num_embeddings_per_table)
    print(sum(num_embeddings_per_table))
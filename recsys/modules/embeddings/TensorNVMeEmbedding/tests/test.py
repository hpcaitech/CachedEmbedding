import sys
from urllib import request
import torch
sys.path.append("../..")

from NVMeWeight.weight_tensor_loader import NVMeWeight
from generator import generate_freq
import os


freq = generate_freq(100)
nvme_weight = NVMeWeight(cpu_row_num = 10, weight_path="weight_folder")
nvme_weight.reorder(freq)

print(nvme_weight.idx_map)

# start
requires = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).to(torch.int32)
nvme_weight.prepare_ids(requires)
for id in requires:
    print(nvme_weight.get_embedding_from_original_id(id))

requires = torch.Tensor([10,1,2,3,4,5,6,7,8,9]).to(torch.int32)
nvme_weight.prepare_ids(requires)
for id in requires:
    print(nvme_weight.get_embedding_from_original_id(id))

requires = torch.Tensor([10,11,12,13,14,15,16,17,18,19]).to(torch.int32)
nvme_weight.prepare_ids(requires)
for id in requires:
    print(nvme_weight.get_embedding_from_original_id(id))

requires = torch.Tensor([99]).to(torch.int32)
nvme_weight.prepare_ids(requires)
for id in requires:
    print(nvme_weight.get_embedding_from_original_id(id))

# print(nvme_weight.get_embedding_from_original_id(70)) # segmentation fault


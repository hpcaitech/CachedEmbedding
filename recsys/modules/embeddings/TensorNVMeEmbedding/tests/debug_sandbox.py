# USELESS FILE

from tensornvme import DiskOffloader
import torch
import os
import numpy as np
from typing import List, Optional, Tuple

#
#temp_weight = np.load("weight_folder/001.npy")
#loader = DiskOffloader(".", backend='aio')
#
#weight :List[torch.Tensor] = []
#for i in range(np.shape(temp_weight)[0]):
#    weight.append(torch.from_numpy(temp_weight[i]).clone())
#
#l = np.random.randn(10)
#a = [torch.from_numpy(l).clone() for i in range(10)]
#print(weight[0].storage())
#print(a[0].storage())
#a[0].storage().resize_(0)
#weight[0].storage().resize_(0)
#loader.async_write(weight[0])

nvme_row_idxs = torch.Tensor([12,42,3,86,22,9,7,38]).to(torch.int32)

cached_idx_map = torch.Tensor([0,1,2,3,4,5,6,7,8,9]).to(torch.int32)

a = torch.isin(nvme_row_idxs, cached_idx_map, invert=True)

print(a)

comm_nvme_row_idxs = nvme_row_idxs[a]

print(comm_nvme_row_idxs)
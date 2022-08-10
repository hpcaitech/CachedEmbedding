import chunk
from typing import List, Optional, Tuple
from tensornvme import DiskOffloader
import torch
import os
import numpy as np
from contexttimer import Timer

from torch.profiler import record_function

class NVMeWeight():
    '''
    A tensor may be too big to be fully loaded on memory.
    To be specific, weight, a 2d tensor, should be sharded by first demension to a list, most of whose tensors stored in disk. 
    Whenever to be visited, load tensors into memory.
    DiskOffloader manage the io.

    How to prepare weights: save weights to several files under a folder.  Files' name should be in dict order.
    NOTE: weights should already been sorted by frequency if enabled frequency strategy
    '''
    def __init__(self,
                cpu_row_num: int = 0,
                weight_path: str = ".",
                loader_backend: str = 'aio'
                ):
        self.loader = DiskOffloader(".", backend=loader_backend)
        self.weight: List[torch.Tensor] = []
        self.cpu_row_num = cpu_row_num

        self.weight_shape = [0,0]
        self._load_weights(weight_path)
        self.num_embeddings = self.weight_shape[0]
        self.embedding_dim = self.weight_shape[1]
        

        self.idx_map = torch.arange(self.num_embeddings, dtype=torch.long) # original id -> sorted weight id (disk id)
        self.cached_idx_map = torch.empty(self.cpu_row_num, dtype=torch.long).fill_(-1)# cpu id -> disk id
        self.inverted_cached_idx = torch.empty(self.num_embeddings, dtype=torch.long).fill_(-1) # disk id -> cpu id

        self.evict_backlist = torch.tensor([])

        self.num_hits_history = []
        self.num_miss_history = []
        self.num_write_back_history = []
        self.input_id_percent_in_load_chunk = []

        self._cpu_available_row_num = cpu_row_num

        self._reset_comm_stats()

    def _load_weights(self, weight_path: str = "."):
        '''
        load weights from files. Shard to list each row
        '''
        files_names = os.listdir(weight_path)
        files_names.sort()
        for each in files_names:
            temp_weight = np.load(os.path.join(weight_path, each))
            self.weight_shape[0] += np.shape(temp_weight)[0]
            for i in range(np.shape(temp_weight)[0]):
                self.weight.append(torch.from_numpy(temp_weight[i]).clone())
                self.loader.sync_write(self.weight[-1])

        # load weight[0] temporary for some info 
        self.loader.sync_read(self.weight[0])
        self.weight_shape[1] = self.weight[0].shape[0]
        self.elem_size_in_byte = self.weight[0].element_size()
        self.loader.sync_write(self.weight[0])

    def _reset_comm_stats(self):
        self._nvme_to_cpu_numel = 0
        self._nvme_to_cpu_elpase = 0
        self._cpu_to_nvme_elapse = 0
        self._cpu_to_nvme_numel = 0

    @property
    def cpu_available_row_num(self):
        return self._cpu_available_row_num

    @torch.no_grad()
    def reorder(self, ids_freq_mapping: Optional[List[int]] = None, warmup_ratio=0.7):
        '''
        reorder the weight according to ids' frequency in dataset before training.
        Also Build the IndexMappingTable, aka index_mapping_table.
        Execute only once before training.
        Args:
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq. if None no reorder
            warmup_ratio (float): the amount of chunks preloaded in cpu cache
        '''
        if ids_freq_mapping is not None:
            tmp_idx = torch.argsort(torch.from_numpy(ids_freq_mapping), descending=True)
            sorted_idx = torch.argsort(tmp_idx)
            self.idx_map.data.copy_(sorted_idx)

        preload_row_num = min(int(np.ceil(self.cpu_row_num * warmup_ratio)), self.num_embeddings)
        if preload_row_num > 0:
            with Timer() as timer:
                # extrack chunks from NVMe weight
                # self.loader.sync_readv(self.weight[0:preload_row_num]) # it doesn't work
                preload_slot_ids = torch.arange(preload_row_num)
                for id in preload_slot_ids:
                    self.loader.sync_read(self.weight[id])

                # update auxiliary info
                
                slot_offsets = preload_slot_ids
                self.cached_idx_map[preload_slot_ids] = preload_slot_ids
                self.inverted_cached_idx[preload_slot_ids] = slot_offsets
                self._cpu_available_row_num -= preload_row_num
            print(f'Cache warmup finished cost {timer.elapsed} sec.')

    def flush(self):
        '''
        flush all memory chunks to disk
        The function is usually called after training finished.
        '''
        slots = torch.nonzero(self.cached_idx_map > -1).squeeze(1)
        chunk_ids = self.cached_idx_map[slots]
        for id in chunk_ids:
            self.loader.sync_write(self.weight[id])
        self.cached_idx_map.index_fill_(0, slots, -1)
        self.inverted_cached_idx.index_fill_(0, chunk_ids, -1)
        self._cpu_available_row_num += slots.numel()

        assert self._cpu_available_row_num == self.cpu_row_num
        assert torch.all(self.inverted_cached_idx == -1).item()
        assert torch.all(self.cached_idx_map == -1).item()

    def print_comm_stats(self):
        if self._cpu_to_nvme_numel > 0:
            print(
                f"CPU->NVMe BWD {self._cpu_to_nvme_numel * self.elem_size_in_byte / 1e6 / self._cpu_to_nvme_elapse} MB/s {self._cpu_to_nvme_numel / 1e6} M elem"
            )
        if self._nvme_to_cpu_numel > 0:
            print(
                f"NVMe->CPU BWD {self._nvme_to_cpu_numel * self.elem_size_in_byte / 1e6 / self._nvme_to_cpu_elapse} MB/s {self._nvme_to_cpu_numel / 1e6} M elem"
            )
        
    @torch.no_grad()
    def _id_to_cached_cpu_id(self, ids: torch.Tensor) -> torch.Tensor:
        '''
        convert ids to indices in cpu weight(cached cpu id)
        Implemented with parallel operations on CPU.

        Args:
            ids (torch.Tensor): ids from the dataset

        Returns:
            torch.Tensor: contains indices in cpu weight
        '''
        ids = self.idx_map.index_select(0,ids.view(-1))
        ret = self.inverted_cached_idx.index_select(0, ids)
        return ret

    @torch.no_grad()
    def prepare_ids(self, ids: torch.Tensor) -> torch.Tensor:
        '''
        move the disk embedding rows w.r.t ids into cpu memory 
        Args:
            ids (torch.Tensor): the ids to be computed
        Returns:
            torch.Tensor: indices on the cpu weight.
        '''
        with record_function("(zhg) get unique indices"):
            nvme_row_idxs = torch.unique(self.idx_map.index_select(0, ids))

            assert len(nvme_row_idxs) <= self.cpu_row_num, \
                f"the input indices pull {len(nvme_row_idxs)} rows, " \
                f"which is larger than the presented {self.cpu_row_num}, " \
                f"please increase cpu_row_num  shrink batch size"
            self.evict_backlist = nvme_row_idxs

        with record_function("(zhg) get nvme indices"):
            # ids need to load, ignore already loaded ids
            comm_nvme_row_idxs = nvme_row_idxs[torch.isin(nvme_row_idxs, self.cached_idx_map, invert=True)]
        
        self.num_hits_history.append(len(nvme_row_idxs) - len(comm_nvme_row_idxs))
        self.num_miss_history.append(len(comm_nvme_row_idxs))
        self.num_write_back_history.append(0)

        # move sure the cpu row will not be evicted!
        with record_function("(zhg) cache update"):
            self._prepare_rows_on_cpu(comm_nvme_row_idxs)

        self.evict_backlist = torch.tensor([], dtype=nvme_row_idxs.dtype)
        
        with record_function("(zhg) embed idx -> cache chunk id"):
            cpu_row_idxs = self._id_to_cached_cpu_id(ids)
        return cpu_row_idxs
    
    def _row_in_cpu(self, row_id: int) -> bool:
        return self.inverted_cached_idx[row_id] != -1

    @torch.no_grad()
    def _prepare_rows_on_cpu(self, nvme_row_idxs: torch.Tensor) -> None:
        '''
        prepare rows in nvme_row_idxs on cpu memory
        Args:
            nvme_row_idxs (torch.Tensor): the rows to be loaded on cpu memory
        '''
        evict_num = nvme_row_idxs.numel() - self.cpu_available_row_num
        if evict_num > 0:
            with Timer() as timer:
                mask_nvme_row_idx = torch.isin(self.cached_idx_map, self.evict_backlist)
                backup_idxs = self.cached_idx_map[mask_nvme_row_idx].clone()
                invalid_idx = torch.nonzero(mask_nvme_row_idx).squeeze(1)

                self.cached_idx_map.index_fill_(0, invalid_idx, -2)
                evict_cpu_row_idx = torch.argsort(self.cached_idx_map, descending=True)[:evict_num]
                self.cached_idx_map.index_copy_(0, invalid_idx, backup_idxs)
                evict_info = self.cached_idx_map[evict_cpu_row_idx]
                # cpu -> nvme
                for id in evict_info:
                    self.loader.sync_write(self.weight[id])

                self.cached_idx_map.index_fill_(0,evict_cpu_row_idx, -1)
                self.inverted_cached_idx.index_fill_(0,evict_info,-1)
                self._cpu_available_row_num += evict_num

                weight_size = evict_cpu_row_idx.numel() * self.embedding_dim
            
            self._cpu_to_nvme_elapse += timer.elapsed
            self._cpu_to_nvme_numel += weight_size
        
        with Timer() as timer:
            slots = torch.nonzero(self.cached_idx_map == -1).squeeze(1)[:nvme_row_idxs.numel()]
            # nvme -> cpu
            for id in nvme_row_idxs:
                self.loader.sync_read(self.weight[id])

            slot_offsets = slots
            self.cached_idx_map[slots] = nvme_row_idxs
            self.inverted_cached_idx.index_copy_(0, nvme_row_idxs, slot_offsets)
            self._cpu_available_row_num -= nvme_row_idxs.numel()
        self._nvme_to_cpu_elpase += timer.elapsed
        weight_size = nvme_row_idxs.numel() * self.embedding_dim
        self._nvme_to_cpu_numel += weight_size
    
    def _evict(self) -> int:
        '''
        evict one row from cpu to nvme.
        Returns:
        (int) : the slot id be evicted.
        '''
        mask = torch.logical_or(torch.isin(self.cached_idx_map, self.evict_backlist), self.cached_idx_map == -1)
        buf = self.cached_idx_map[mask].clone()
        idx = torch.nonzero(mask).squeeze(1)
        self.cached_idx_map.index_fill_(0, idx, -1)
        max_row, max_nvme_row_idx = torch.max(self.cached_idx_map, dim=0)
        max_cpu_row_idx = self.cached_idx_map[max_nvme_row_idx]

        if max_cpu_row_idx == -1:
            raise RuntimeError("Can not evict a row")

        max_cpu_row_idx = max_cpu_row_idx.item()
        max_offset = self.inverted_cached_idx[max_cpu_row_idx]
        # recover
        self.cached_idx_map.index_copy_(0, idx, buf)

        with Timer() as timer:
            self.loader.sync_write(self.weight[max_cpu_row_idx])

        # update inverted_cached_idx, min_slot_id is evicted from cpu
        self.cached_idx_map[max_nvme_row_idx] = -1

        self.inverted_cached_idx[max_cpu_row_idx] = -1

        self._cpu_available_row_num += 1

        self._cpu_to_nvme_numel += self.embedding_dim
        self._cpu_to_nvme_elapse += timer.elapsed
        # self.num_write_back_history[-1] += 1
        return max_nvme_row_idx

    def _find_free_cpu_row(self) -> int:
        if self._cpu_available_row_num == 0:
            return -1
        candidates = torch.nonzero(self.cached_idx_map == -1).squeeze(1)
        return candidates[0].item()

    @torch.no_grad()
    def _admit(self, row_id: int):
        """
        move in row_id to cpu

        Args:
            row_id (int): the id of row to be moved in
        """
        # find a free slot in partial cuda weight
        slot_id = self._find_free_cpu_row()

        if slot_id == -1:
            # evict one row
            slot_id = self._evict()
        slot_offset = slot_id
        # copy payload from nvme to cpu
        with Timer() as timer:
            self.loader.sync_read(self.weight[row_id])

        # update the inverted_cached_idx
        self.cached_idx_map[slot_id] = row_id
        self.inverted_cached_idx[row_id] = slot_offset

        self._cpu_available_row_num -= 1

        self._nvme_to_cpu_numel += self.embedding_dim
        self._nvme_to_cpu_elpase += timer.elapsed

    def get_embedding_from_original_id(self, id:int):
        return self.weight[self.idx_map[id]]
import numpy as np
import torch
from torch.profiler import record_function
from typing import List, Optional
from contexttimer import Timer


class ChunkParamMgr(object):
    """
    Manage Chunk Weights on CPU and CUDA memory.
    CPU maintains a replica of the original weight. CUDA maintains a subset of weight chunks.
    During training, we need to swapin/out chunks.
    """

    def __init__(self,
                 weight: torch.Tensor,
                 chunk_size: int = 16 * 1024 * 1024,
                 cuda_chunk_num: int = 0,
                 *args,
                 **kwargs) -> None:
        self.chunk_size = chunk_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_chunk_num = cuda_chunk_num
        self._cuda_available_chunk_num = cuda_chunk_num

        self.elem_size_in_byte = weight.element_size()

        self.cuda_partial_weight = torch.nn.Parameter(
            torch.zeros(cuda_chunk_num * chunk_size, self.embedding_dim, device=torch.cuda.current_device()))

        self.chunk_num = (self.num_embeddings + chunk_size - 1) // chunk_size

        if weight.device.type == 'cuda':
            weight = weight.cpu()

        # Padding the weight to handle cases where `num_embeddings` is not divisible by chunk_size
        mod = weight.shape[0] % chunk_size
        if mod > 0:
            with torch.no_grad():
                padding = torch.zeros(chunk_size - mod, weight.shape[1], device=weight.device, dtype=weight.dtype)
                weight = torch.cat([weight, padding], dim=0)

        # pin memory cpu for higher CPU-GPU copy bandwidth
        self.cpu_weight = weight.pin_memory()

        # IndexMappingTable (IMP): implemented with two lists. 
        # id-> chunk_id and id -> offset_in_chunk
        # It is a static table build by reorder and never changes during training

        # id -> chunk_id
        self.IMP_chunkid = torch.arange(self.num_embeddings, dtype=torch.long,
                                        device=torch.cuda.current_device()).unsqueeze(1)
        # id -> offset_in_chunk
        self.IMP_offsetinchunk = torch.arange(self.num_embeddings, dtype=torch.long,
                                              device=torch.cuda.current_device()).unsqueeze(1)

        # CachedChunkTable: dict(slot_idx, (chunk_id, offset)), slot_ids is the offset in Tensor self.cuda_partial_weight
        self.cached_chunk_table = torch.empty(cuda_chunk_num, 2, device=torch.cuda.current_device(),
                                              dtype=torch.long).fill_(-1)

        # chunk_id, slot_offset. slot_offset is the offset in chunk, -1 means chunk_id not in CUDA.
        self.CCT = torch.zeros(self.chunk_num, 1, device=torch.cuda.current_device(), dtype=torch.long).fill_(-1)

        self.evict_backlist = torch.tensor([], device = torch.cuda.current_device())

        self.num_hits_history = []
        self.num_miss_history = []
        self.num_write_back_history = []
        self.input_id_percent_in_load_chunk = []
        self._reset_comm_stats()

    def cpu_weight_chunk(self, chunk_id: int) -> torch.Tensor:
        """
        access a chunk of CPU weight.

        Args:
            chunk_id (int): chunk id

        Returns:
            torch.Tensor: a piece of memory in CPU weight corresponding to chunk id's payload. The tensor is 1-D.
        """

        return self.cpu_weight.data.view(-1).narrow(0,
                                                    int(chunk_id) * self.chunk_size * self.embedding_dim,
                                                    self.chunk_size * self.embedding_dim).view(
                                                        self.chunk_size, self.embedding_dim)

    def _reset_comm_stats(self):
        self._cpu_to_cuda_numel = 0
        self._cpu_to_cuda_elpase = 0
        self._cuda_to_cpu_elapse = 0
        self._cuda_to_cpu_numel = 0

    def _chunk_in_cuda(self, chunk_id : int) -> bool:
        return self.CCT[chunk_id] != -1

    @property
    def cuda_available_chunk_num(self):
        return self._cuda_available_chunk_num

    @torch.no_grad()
    def reorder(self, ids_freq_mapping: Optional[List[int]] = None):
        """reorder the cpu_weight according to ids' frequency in dataset before training.
        Also Build the IndexMappingTable, aka index_mapping_table.
        Execute only once before training.
        Args:
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq. if None no reorder
        """
        if ids_freq_mapping is not None:
            sorted_idx = torch.argsort(torch.from_numpy(ids_freq_mapping).cuda(), descending=True)
        else:
            sorted_idx = torch.arange(self.num_embeddings, device=torch.cuda.current_device(), dtype=torch.long)

        divs = torch.div(sorted_idx, self.chunk_size, rounding_mode='floor').unsqueeze(1)
        mods = torch.remainder(sorted_idx, self.chunk_size).unsqueeze(1)

        self.IMP_chunkid.data.copy_(divs)
        self.IMP_offsetinchunk.data.copy_(mods)

    @torch.no_grad()
    def _id_to_cached_cuda_id(self, ids: torch.Tensor) -> torch.Tensor:
        """
        convert ids to indices in self.partial_cuda_weight.
        Implemented with parallel operations on GPU.

        Args:
            ids (torch.Tensor): ids from the dataset

        Returns:
            torch.Tensor: contains indices in self.partial_cuda_weight
        """
        ids = ids.view(-1)
        chunk_ids = self.IMP_chunkid.index_select(0, ids)
        offset_in_chunks = self.IMP_offsetinchunk.index_select(0, ids)
        ret = self.CCT.index_select(0, chunk_ids.view(-1)) + offset_in_chunks
        return ret

    @torch.no_grad()
    def prepare_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        move the chunks w.r.t. ids into CUDA memory

        Args:
            ids (torch.Tensor): the ids to be computed
        Returns:
            torch.Tensor: indices on the cuda_partial_weight.
        """
        with record_function("(zhg) get unique indices"):
            # unique(IMT(ids)) -> chunk ids
            chunk_id_set = torch.unique(self.IMP_chunkid.index_select(0, ids))

            assert len(chunk_id_set) <= self.cuda_chunk_num, \
                f"the input indices pull {len(chunk_id_set)} chunks, " \
                f"which is larger than the presented {self.cuda_chunk_num}, " \
                f"please increase cuda_chunk_num and chunk_size or shrink batch size"
            self.evict_backlist = chunk_id_set

        with record_function("(zhg) get cpu chunk indices"):
            cpu_chunk_id_list = chunk_id_set[torch.isin(chunk_id_set, self.cached_chunk_table[:, 0],
                                                        invert=True)].tolist()

        self.num_hits_history.append(len(chunk_id_set) - len(cpu_chunk_id_list))
        self.num_miss_history.append(len(cpu_chunk_id_list))
        self.num_write_back_history.append(0)

        # move sure the cuda chunk will not be evicted!
        with record_function("(zhg) cache update"):
            self._prepare_chunks_on_cuda(cpu_chunk_id_list)

        self.evict_backlist = torch.tensor([], device=chunk_id_set.device, dtype=chunk_id_set.dtype)
        # new ids chunk_offset + offset_in_chunk
        with record_function("(zhg) embed idx -> cache chunk id"):
            mapped_ids = self._id_to_cached_cuda_id(ids).view(ids.shape)
        return mapped_ids

    def _prepare_chunks_on_cuda(self, chunk_ids: List[int]) -> None:
        """prepare chunks in chunk_ids on CUDA memory
        Args:
            chunk_ids (List[int]): the chunks to be placed on CUDA
        """
        for chunk_id in chunk_ids:
            self._admit(chunk_id)

    def _evict(self) -> int:
        """
        evict one chunk from cuda to cpu.
        Returns: 
        (int) : the slot id be evicted.
        """
        # min_chunk_id = 2 * self.chunk_num
        # min_slot_id = None
        # min_offset = None
        # for slot_id, row in enumerate(self.cached_chunk_table):
        #     if 0 <= row[0] < min_chunk_id and row[0] not in self.evict_backlist:
        #         min_chunk_id = row[0].item()
        #         min_slot_id = slot_id
        #         min_offset = row[1].item()
        #
        # if min_slot_id is None:
        #     raise RuntimeError("Can not evict a chunk")
        max_int_value = 2147483647

        mask = torch.logical_or(torch.isin(self.cached_chunk_table[:, 0], self.evict_backlist),
                                self.cached_chunk_table[:, 0] == -1)
        buf = self.cached_chunk_table[mask, 0].clone()
        idx = torch.nonzero(mask).squeeze(1)
        self.cached_chunk_table[:, 0].index_fill_(0, idx, max_int_value)
        min_row, min_slot_id = torch.min(self.cached_chunk_table[:, 0], dim=0)

        min_chunk_id, min_offset = self.cached_chunk_table[min_slot_id]
        
        if min_chunk_id == max_int_value:
            raise RuntimeError("Can not evict a chunk")
         
        min_chunk_id = min_chunk_id.item()
        # recover
        self.cached_chunk_table[:, 0].index_copy_(0, idx, buf)


        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_partial_weight.view(-1), 0, min_offset * self.embedding_dim,
                                       self.chunk_size * self.embedding_dim).view(self.chunk_size, self.embedding_dim)
            self.cpu_weight_chunk(min_chunk_id).data.copy_(cuda_tensor)

        # update CCT, min_slot_id is evicted from cuda
        self.cached_chunk_table[min_slot_id, 0] = -1
        self.CCT[min_chunk_id] = -1

        self._cuda_available_chunk_num += 1

        self._cuda_to_cpu_numel += self.chunk_size * self.embedding_dim
        self._cuda_to_cpu_elapse += timer.elapsed
        # self.num_write_back_history[-1] += 1
        return min_slot_id

    def _find_free_cuda_slot(self) -> int:
        if self._cuda_available_chunk_num == 0:
            return -1
        candidates = torch.nonzero(self.cached_chunk_table[:, 0] == -1).squeeze(1)
        return candidates[0].item()

    @torch.no_grad()
    def _admit(self, chunk_id: int):
        """
        move in chunk_id to CUDA

        Args:
            chunk_id (int): the id of chunk to be moved in
        """
        # find a free slot in partial cuda weight
        slot_id = self._find_free_cuda_slot()

        if slot_id == -1:
            # evict one chunk
            slot_id = self._evict()

        slot_offset = slot_id * self.chunk_size
        # copy payload from cpu to cuda
        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_partial_weight.view(-1), 0, slot_offset * self.embedding_dim,
                                       self.chunk_size * self.embedding_dim).view(self.chunk_size, self.embedding_dim)
            cuda_tensor.data.copy_(self.cpu_weight_chunk(chunk_id))

        # update the CCT
        self.cached_chunk_table[slot_id].data.copy_(
            torch.tensor((chunk_id, slot_offset), device=torch.cuda.current_device(), dtype=torch.float32))
        self.CCT[chunk_id] = slot_offset

        self._cuda_available_chunk_num -= 1

        self._cpu_to_cuda_numel += self.chunk_size * self.embedding_dim
        self._cpu_to_cuda_elpase += timer.elapsed

    def flush(self):
        """flush all CUDA chunks to CPU.
        The function is usually called after training finished.
        """
        while self._cuda_available_chunk_num < self.cuda_chunk_num:
            self._evict()

    def print_comm_stats(self):
        if self._cuda_to_cpu_numel > 0:
            print(
                f"CUDA->CPU BWD {self._cuda_to_cpu_numel * self.elem_size_in_byte / 1e6 / self._cuda_to_cpu_elapse} MB/s {self._cuda_to_cpu_numel / 1e6} M elem"
            )
        if self._cpu_to_cuda_numel > 0:
            print(
                f"CPU->CUDA BWD {self._cpu_to_cuda_numel * self.elem_size_in_byte / 1e6 / self._cpu_to_cuda_elpase} MB/s {self._cpu_to_cuda_numel / 1e6} M elem"
            )

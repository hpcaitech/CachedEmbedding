import numpy as np
import torch
from torch.profiler import record_function
from typing import List, Optional
from contexttimer import Timer
from .limit_buff_index_copy import LimitBuffIndexCopyer


class ChunkParamMgr(torch.nn.Module):
    """
    Manage Weights in Chunk on CPU and CUDA memory.
    CPU maintains a replica of the original weight. CUDA maintains a subset of weight chunks used in the comming computation.
    During training, GPU needs to admit/evict chunks.
    """

    def __init__(self,
                 weight: torch.Tensor,
                 chunk_size: int = 1,
                 cuda_row_num: int = 0,
                 buffer_size: int = 50_000) -> None:
        super(ChunkParamMgr, self).__init__()
        self.buffer_size = buffer_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_row_num = cuda_row_num
        self._cuda_available_row_num = self.cuda_row_num

        self.elem_size_in_byte = weight.element_size()

        self.cuda_cached_weight = torch.nn.Parameter(
            torch.zeros(self.cuda_row_num,
                        self.embedding_dim,
                        device=torch.cuda.current_device(),
                        dtype=weight.dtype))


        if weight.device.type == 'cuda':
            weight = weight.cpu()

        # pin memory cpu for higher CPU-GPU copy bandwidth
        self.cpu_weight = weight.contiguous().pin_memory()

        # map original id to new id with respect to frequency
        # id -> cpu_row_idx
        self.register_buffer(
            "idx_map",
            torch.arange(self.num_embeddings, dtype=torch.long, device=torch.cuda.current_device()),
            persistent=False,
        )

        # CachedChunkTable: gpu_row_idx -> cpu_row_idx
        self.register_buffer("cached_idx_map",
                             torch.empty(self.cuda_row_num, device=torch.cuda.current_device(),
                                         dtype=torch.long).fill_(-1),
                             persistent=False)

        # cpu_row_id -> gpu_row_idx.
        # gpu_row_idx as -1 means cpu_row_id not in CUDA.
        self.register_buffer("inverted_cached_idx",
                             torch.zeros(self.num_embeddings, device=torch.cuda.current_device(),
                                         dtype=torch.long).fill_(-1),
                             persistent=False)

        self.evict_backlist = torch.tensor([], device=torch.cuda.current_device())

        # index copy buffer size should less than 10% of cuda weight.
        if self.buffer_size > 0:
            self.limit_buff_index_copyer = LimitBuffIndexCopyer(self.buffer_size)

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
                                                    int(chunk_id) * self.embedding_dim,
                                                    self.embedding_dim).view(
                                                        1, self.embedding_dim)

    @property
    def cuda_available_chunk_num(self):
        return self._cuda_available_row_num

    @torch.no_grad()
    def reorder(self, ids_freq_mapping: Optional[List[int]] = None, warmup_ratio=0.7):
        """reorder the cpu_weight according to ids' frequency in dataset before training.
        Also Build the IndexMappingTable, aka index_mapping_table.
        Execute only once before training.
        Args:
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq. if None no reorder
            warmup_ratio (float): the amount of chunks preloaded in cuda cache
        """
        if ids_freq_mapping is not None:
            tmp_idx = torch.argsort(torch.from_numpy(ids_freq_mapping).cuda(), descending=True)
            sorted_idx = torch.argsort(tmp_idx)
            self.idx_map.data.copy_(sorted_idx)

        # TODO() The following code will allocate extra CUDA memory. preload_chunk_num * chunks.
        # As cuda_cached_weight is very big. You may not have that much available memory!
        # Warmup the cuda cache by moving high freq chunks (lowest chunk id) to cuda
        preload_chunk_num = min(int(np.ceil(self.cuda_row_num * warmup_ratio)), self.num_embeddings)
        if preload_chunk_num > 0:
            with Timer() as timer:
                # extract chunks from cpu weight
                preload_chunk_ids = torch.arange(preload_chunk_num)
                preload_slot_ids = preload_chunk_ids.cuda()

                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(0,
                                                            src_index=preload_chunk_ids,
                                                            tgt_index=preload_slot_ids,
                                                            src=self.cpu_weight.view(self.num_embeddings, -1),
                                                            tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1))
                else:
                    preload_chunks = self.cpu_weight.view(self.num_embeddings, -1).index_select(0, preload_chunk_ids).cuda()
                    self.cuda_cached_weight.view(self.cuda_row_num,
                                                  -1).index_copy_(0, preload_slot_ids, preload_chunks)

                # update auxiliary info
                slot_offsets = preload_slot_ids
                self.cached_idx_map[preload_slot_ids] = preload_slot_ids
                self.inverted_cached_idx[preload_slot_ids] = slot_offsets
                self._cuda_available_row_num -= preload_chunk_num
            print(f'Cache warmup finished cost {timer.elapsed} sec.')

    def flush(self):
        """flush all CUDA chunks to CPU.
        The function is usually called after training finished.
        """
        slots = torch.nonzero(self.cached_idx_map > -1).squeeze(1)
        chunk_ids = self.cached_idx_map[slots]
        chunks = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(0, slots).cpu()
        self.cpu_weight.view(self.num_embeddings, -1).index_copy_(0, chunk_ids.cpu(), chunks)
        self.cached_idx_map.index_fill_(0, slots, -1)
        self.inverted_cached_idx.index_fill_(0, chunk_ids, -1)
        self._cuda_available_row_num += slots.numel()

        assert self._cuda_available_row_num == self.cuda_row_num
        assert torch.all(self.inverted_cached_idx == -1).item()
        assert torch.all(self.cached_idx_map == -1).item()

    def print_comm_stats(self):
        if self._cuda_to_cpu_numel > 0:
            print(
                f"CUDA->CPU BWD {self._cuda_to_cpu_numel * self.elem_size_in_byte / 1e6 / self._cuda_to_cpu_elapse} MB/s {self._cuda_to_cpu_numel / 1e6} M elem"
            )
        if self._cpu_to_cuda_numel > 0:
            print(
                f"CPU->CUDA BWD {self._cpu_to_cuda_numel * self.elem_size_in_byte / 1e6 / self._cpu_to_cuda_elpase} MB/s {self._cpu_to_cuda_numel / 1e6} M elem"
            )

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
        ids = self.idx_map.index_select(0, ids.view(-1))
        ret = self.inverted_cached_idx.index_select(0, ids)
        return ret

    @torch.no_grad()
    def prepare_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        move the chunks w.r.t. ids into CUDA memory

        Args:
            ids (torch.Tensor): the ids to be computed
        Returns:
            torch.Tensor: indices on the cuda_cached_weight.
        """
        with record_function("(zhg) get unique indices"):
            # unique(IMT(ids)) -> chunk ids
            cpu_row_idxs = torch.unique(self.idx_map.index_select(0, ids))

            assert len(cpu_row_idxs) <= self.cuda_row_num, \
                f"the input indices pull {len(cpu_row_idxs)} chunks, " \
                f"which is larger than the presented {self.cuda_row_num}, " \
                f"please increase cuda_row_num  shrink batch size"
            self.evict_backlist = cpu_row_idxs

        with record_function("(zhg) get cpu chunk indices"):
            cpu_chunk_id_list = cpu_row_idxs[torch.isin(cpu_row_idxs, self.cached_idx_map, invert=True)]

        self.num_hits_history.append(len(cpu_row_idxs) - len(cpu_chunk_id_list))
        self.num_miss_history.append(len(cpu_chunk_id_list))
        self.num_write_back_history.append(0)

        # move sure the cuda chunk will not be evicted!
        with record_function("(zhg) cache update"):
            self._prepare_chunks_on_cuda(cpu_chunk_id_list)

        self.evict_backlist = torch.tensor([], device=cpu_row_idxs.device, dtype=cpu_row_idxs.dtype)
        # new ids chunk_offset + offset_in_chunk
        with record_function("(zhg) embed idx -> cache chunk id"):
            ret = self._id_to_cached_cuda_id(ids)
        return ret

    def _reset_comm_stats(self):
        self._cpu_to_cuda_numel = 0
        self._cpu_to_cuda_elpase = 0
        self._cuda_to_cpu_elapse = 0
        self._cuda_to_cpu_numel = 0

    def _chunk_in_cuda(self, chunk_id: int) -> bool:
        return self.inverted_cached_idx[chunk_id] != -1

    @torch.no_grad()
    def _prepare_chunks_on_cuda(self, chunk_ids: torch.Tensor) -> None:
        """prepare chunks in chunk_ids on CUDA memory
        Args:
            chunk_ids (torch.Tensor): the chunks to be placed on CUDA
        """
        evict_num = chunk_ids.numel() - self.cuda_available_chunk_num
        if evict_num > 0:
            with Timer() as timer:
                mask = torch.isin(self.cached_idx_map, self.evict_backlist)
                buf = self.cached_idx_map[mask].clone()
                idx = torch.nonzero(mask).squeeze(1)

                self.cached_idx_map.index_fill_(0, idx, -2)
                evict_slot_ids = torch.argsort(self.cached_idx_map, descending=True)[:evict_num]
                self.cached_idx_map.index_copy_(0, idx, buf)

                evict_info = self.cached_idx_map[evict_slot_ids]

                if self.buffer_size > 0:
                    self.limit_buff_index_copyer.index_copy(0,
                                                            src_index=evict_slot_ids,
                                                            tgt_index=evict_info.cpu(),
                                                            src=self.cuda_cached_weight.view(self.cuda_row_num, -1),
                                                            tgt=self.cpu_weight.view(self.num_embeddings, -1))
                else:
                    # allocate tmp memory on CPU and copy chunks on CUDA to CPU.
                    chunks = self.cuda_cached_weight.view(self.cuda_row_num, -1).index_select(0,
                                                                                                 evict_slot_ids).cpu()
                    self.cpu_weight.view(self.num_embeddings, -1).index_copy_(0, evict_info.cpu(), chunks)

                self.cached_idx_map.index_fill_(0, evict_slot_ids, -1)
                self.inverted_cached_idx.index_fill_(0, evict_info, -1)
                self._cuda_available_row_num += evict_num

                weight_size = evict_slot_ids.numel() * self.embedding_dim
            self._cuda_to_cpu_elapse += timer.elapsed
            self._cuda_to_cpu_numel += weight_size
            # print(f"evict embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

        with Timer() as timer:
            slots = torch.nonzero(self.cached_idx_map == -1).squeeze(1)[:chunk_ids.numel()]
            # Here also allocate extra memory on CUDA. #chunk_ids * chunk.
            if self.buffer_size > 0:
                self.limit_buff_index_copyer.index_copy(0,
                                                        src_index=chunk_ids.cpu(),
                                                        tgt_index=slots,
                                                        src=self.cpu_weight.view(self.num_embeddings, -1),
                                                        tgt=self.cuda_cached_weight.view(self.cuda_row_num, -1))
            else:
                chunks = self.cpu_weight.view(self.num_embeddings, -1).index_select(0, chunk_ids.cpu()).cuda()
                self.cuda_cached_weight.view(self.cuda_row_num, -1).index_copy_(0, slots, chunks)
            slot_offsets = slots
            self.cached_idx_map[slots] = chunk_ids
            self.inverted_cached_idx.index_copy_(0, chunk_ids, slot_offsets)
            self._cuda_available_row_num -= chunk_ids.numel()
        self._cpu_to_cuda_elpase += timer.elapsed
        weight_size = chunk_ids.numel() * self.embedding_dim
        self._cpu_to_cuda_numel += weight_size
        # print(f"admit embedding weight: {weight_size*self.elem_size_in_byte/1e6:.2f} MB")

    def _evict(self) -> int:
        """
        evict one chunk from cuda to cpu.
        Returns: 
        (int) : the slot id be evicted.
        """
        mask = torch.logical_or(torch.isin(self.cached_idx_map, self.evict_backlist), self.cached_idx_map == -1)
        buf = self.cached_idx_map[mask].clone()
        idx = torch.nonzero(mask).squeeze(1)
        self.cached_idx_map.index_fill_(0, idx, -1)
        max_row, max_slot_id = torch.max(self.cached_idx_map, dim=0)
        # print(f"evict: {max_slot_id}")
        max_chunk_id = self.cached_idx_map[max_slot_id]

        if max_chunk_id == -1:
            raise RuntimeError("Can not evict a chunk")

        max_chunk_id = max_chunk_id.item()
        max_offset = self.inverted_cached_idx[max_chunk_id]
        # recover
        self.cached_idx_map.index_copy_(0, idx, buf)

        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_cached_weight.view(-1), 0, max_offset * self.embedding_dim,
                                       self.embedding_dim).view(1, self.embedding_dim)
            self.cpu_weight_chunk(max_chunk_id).data.copy_(cuda_tensor)

        # update inverted_cached_idx, min_slot_id is evicted from cuda
        self.cached_idx_map[max_slot_id] = -1

        self.inverted_cached_idx[max_chunk_id] = -1

        self._cuda_available_row_num += 1

        self._cuda_to_cpu_numel += self.embedding_dim
        self._cuda_to_cpu_elapse += timer.elapsed
        # self.num_write_back_history[-1] += 1
        return max_slot_id

    def _find_free_cuda_slot(self) -> int:
        if self._cuda_available_row_num == 0:
            return -1
        candidates = torch.nonzero(self.cached_idx_map == -1).squeeze(1)
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
        # print(f'admit to {slot_id}, id: {chunk_id}')
        slot_offset = slot_id
        # copy payload from cpu to cuda
        with Timer() as timer:
            cuda_tensor = torch.narrow(self.cuda_cached_weight.view(-1), 0, slot_offset * self.embedding_dim,
                                       self.embedding_dim).view(1, self.embedding_dim)
            cuda_tensor.data.copy_(self.cpu_weight_chunk(chunk_id))

        # update the inverted_cached_idx
        self.cached_idx_map[slot_id] = chunk_id
        self.inverted_cached_idx[chunk_id] = slot_offset

        self._cuda_available_row_num -= 1

        self._cpu_to_cuda_numel += self.embedding_dim
        self._cpu_to_cuda_elpase += timer.elapsed

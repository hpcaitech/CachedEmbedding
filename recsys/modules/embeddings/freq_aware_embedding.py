
import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class ChunkCUDAWeightMgr(object):
    """
    Manage Chunk Weights on CPU and CUDA memory.
    CPU maintains a replica of the original weight. CUDA maintains a subset of weight chunks.
    During training, we need to swapin/out chunks.
    """
    def __init__(self, weight : torch.Tensor, chunk_size : int, cuda_chunk_num : int) -> None:
        self.chunk_size = chunk_size
        self.num_embeddings, self.embedding_dim = weight.shape
        self.cuda_chunk_num = cuda_chunk_num

        self.cuda_partial_weight = torch.empty(cuda_chunk_num * chunk_size * self.embedding_dim, device=torch.cuda.current_device())
        
        self.chunk_num = (self.num_embeddings + chunk_size - 1) // chunk_size
        self.cpu_weight = list(torch.empty(self.chunk_size * self.embedding_dim) for _ in range(self.chunk_num))
    
    def reorder(self, ids_freq_mapping : List[int]):
        """reorder the cpu_weight according to ids' frequency in dataset

        Args:
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
        """
        pass

    def prepare_ids(self, ids: List[int]) -> List[int]:
        """
        move the chunks w.r.t. ids into CUDA memory

        Args:
            ids (List[int]): the ids to be computed
        Returns:
            (List[int]): indices on the cuda_partial_weight.
        """
        pass

    def prepare_cuda_chunks(self, chunk_ids : List[int]) -> None:
        """prepare chunks in chunk_ids on CUDA memory
        Args:
            chunk_ids (List[int]): the chunks to be placed on CUDA
        """
        pass

    def _evict(self, evit_chunk_num : int):
        """
        evict evit_chunk_num chunks from cuda to cpu.

        Args:
            evit_chunk_num (int): the number of chunks to be evicted
        """
        pass

    def _admit(self, chunk_id : int):
        """
        move in chunk_id to CUDA

        Args:
            chunk_id (int): the id of chunk to be moved in
        """
        pass



class FreqAwareEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def preprocess(self, chunk_size : int, cuda_chunk_num : int, ids_freq_mapping : List[int]):
        """
        Called after initialized. 
        Reorder the weight rows according to the ids_freq_mapping.
        Then, let the weights of the Module be managed by a ChunkCUDAWeightMgr.
        Args:
            chunk_size (int): chunk size
            cuda_chunk_num (int): number of chunk can be hosted in CUDA memory
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
        """
        self.chunkweightmgr = ChunkCUDAWeightMgr(self.weight, chunk_size, cuda_chunk_num, ids_freq_mapping)


    def forward(self, indices, offsets=None, per_sample_weights=None):
        reorder_ids = self.chunkweightmgr.prepare_ids(indices)

        embeddings = F.embedding_bag(reorder_ids, self.chunkweightmgr.cuda_partial_weight, offsets, self.max_norm, self.norm_type,
                                     self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                     self.include_last_offset, self.padding_idx)

        return embeddings
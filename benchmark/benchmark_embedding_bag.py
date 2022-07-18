import torch.nn.functional as F
from torch import nn 
import torch
from contexttimer import Timer

def run_embedding_bag(device, num_embeddings, embedding_dim, batch_size):
    embedding_sum = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum').to(device)

    input_ids = torch.randint(low = 0, high = num_embeddings - 1, size = (batch_size, embedding_dim)).to(device)

    output = embedding_sum(input_ids)


def benchmark(num_embeddings, embedding_dim, batch_size):
    with Timer() as cuda_timer:
        for i in range(10):
            run_embedding_bag(torch.cuda.current_device(), num_embeddings, embedding_dim, batch_size)
    
    print('cuda time ', cuda_timer.elapsed)

    with Timer() as cpu_timer:
        for i in range(10):
            run_embedding_bag('cpu', 10000, 2048, 4)
    print('cpu time ', cpu_timer.elapsed)

    print(f'batch_size {batch_size} embedding_dim {embedding_dim} cuda/cpu {cuda_timer.elapsed / cpu_timer.elapsed}')

if __name__ == '__main__':
    for batch_size in [4, 16, 32]:
        for embedding_dim in [128, 1024, 2048]:
            benchmark(10000, embedding_dim, batch_size)


    
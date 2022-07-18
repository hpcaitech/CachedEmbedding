import torch.nn.functional as F
from torch import nn 
import torch
from contexttimer import Timer

def run_embedding_bag(input_ids, output_grad, embedding_sum):
    output = embedding_sum(input_ids)
    output.backward(output_grad)


def benchmark(num_embeddings, embedding_dim, batch_size):
    device = 'cuda'
    embedding_sum = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum').to(device)
    input_ids = torch.randint(low = 0, high = num_embeddings - 1, size = (batch_size, embedding_dim)).to(device)
    output_grad = torch.randn(batch_size, embedding_dim).to(device)

    with Timer() as cuda_timer:
        for i in range(10):
            run_embedding_bag(input_ids, output_grad, embedding_sum)
    
    device = 'cpu'
    embedding_sum = embedding_sum.to(device)
    input_ids = input_ids.to(device)
    output_grad = output_grad.to(device)
    with Timer() as cpu_timer:
        for i in range(10):
            run_embedding_bag(input_ids, output_grad, embedding_sum)

    print(f'batch_size {batch_size} embedding_dim {embedding_dim} cuda/cpu {cuda_timer.elapsed / cpu_timer.elapsed}, cpu time {cpu_timer.elapsed}, cuda time {cuda_timer.elapsed}')

if __name__ == '__main__':
    for batch_size in [4, 16, 32]:
        for embedding_dim in [128, 1024, 2048]:
            benchmark(10000, embedding_dim, batch_size)


    
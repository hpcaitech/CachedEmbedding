#!/bin/bash


# For torchrec's DLRM in baselines
torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script baselines/dlrm_main.py -- --embedding_dim 128  \
    --num_embeddings 3005_0000 \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --learning_rate 1. \
    --batch_size 8192  --limit_train_batches 200 --pin_memory --memory_fraction 0.4

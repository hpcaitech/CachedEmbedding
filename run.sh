#!/bin/bash

# For TorchRec baseline
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script baselines/dlrm_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches --learning_rate 1. --batch_size 8192 #--change_lr

# For Colossal-AI enabled colo_recsys
torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script colo_recsys/dlrm_main.py -- \
    --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches \
    --learning_rate 8e-3 --batch_size 16384

# For PyTorch enabled recsys
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script recsys/dlrm_main.py -- --kaggle \
#    --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory \
#    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches \
#    --learning_rate 1.e-3 --batch_size 16384 --use_sparse_embed_grad --seed 1024
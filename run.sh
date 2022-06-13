#!/bin/bash

# For dlrm_main.py
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script baselines/dlrm_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches --learning_rate 1. --batch_size 8192 #--change_lr

# For colossal
torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script baselines/colossal_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches --learning_rate 1. --batch_size 16384 #--use_cpu #--change_lr

# For hybrid parallel
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script hybrid_parallel_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --shuffle_batches --learning_rate 1. --batch_size 16384 #--use_cpu #--change_lr

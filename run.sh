#!/bin/bash

# For dlrm_main.py
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script dlrm_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --change_lr --shuffle_batches --learning_rate 15.0 --batch_size 8192 --lr_change_point 0.80 --lr_after_change_point 0.20

# For colossal
torchx run -s local_cwd dist.ddp -j 1x2 --script colossal_main.py -- --kaggle --in_memory_binary_criteo_path criteo_kaggle --embedding_dim 128 --pin_memory --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --epochs 1 --change_lr --shuffle_batches --learning_rate 15.0 --batch_size 8192 --lr_change_point 0.80 --lr_after_change_point 0.20
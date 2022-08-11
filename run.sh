#!/bin/bash

# For TorchRec baseline
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script baselines/dlrm_main.py -- \
#    --kaggle --in_memory_binary_criteo_path criteo_kaggle_data --embedding_dim 128 --pin_memory \
#    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
#    --learning_rate 1. --batch_size 8192

# For Colossalai enabled recsys
# criteo kaggle
torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script recsys/dlrm_main.py -- \
    --dataset_dir criteo_kaggle_data --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
     --profile_dir "tensorboard_log/cache"  --buffer_size 0 --use_overlap

# avazu
#torchx run -s local_cwd -cfg log_dir=tmp dist.ddp -j 1x2 --script recsys/dlrm_main.py -- \
#    --dataset_dir avazu_sample --pin_memory --shuffle_batches \
#    --learning_rate 5e-2 --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#    --profile_dir "tensorboard_log/cache"  --buffer_size 0 --use_overlap

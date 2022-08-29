#!/bin/bash

# For TorchRec baseline
torchx run -s local_cwd -cfg log_dir=log/torchrec_avazu/w1_16k dist.ddp -j 1x1 --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path /data/avazu_sample --embedding_dim 128 --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
    --learning_rate 1. --batch_size 16384 --profile_dir "tensorboard_log/torchrec_avazu/w1_16k"

torchx run -s local_cwd -cfg log_dir=log/torchrec_avazu/w2_16k dist.ddp -j 1x2 --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path /data/avazu_sample --embedding_dim 128 --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
    --learning_rate 1. --batch_size 8192  --profile_dir "tensorboard_log/torchrec_avazu/w2_16k"

torchx run -s local_cwd -cfg log_dir=log/torchrec_avazu/w4_16k dist.ddp -j 1x4 --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path /data/avazu_sample --embedding_dim 128 --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
    --learning_rate 1. --batch_size 4096  --profile_dir "tensorboard_log/torchrec_avazu/w4_16k"

torchx run -s local_cwd -cfg log_dir=log/torchrec_avazu/w8_16k dist.ddp -j 1x8 --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path /data/avazu_sample --embedding_dim 128 --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
    --learning_rate 1. --batch_size 2048  --profile_dir "tensorboard_log/torchrec_avazu/w8_16k"
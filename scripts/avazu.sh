#!/bin/bash

# For Colossalai enabled recsys
# avazu
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
torchx run -s local_cwd -cfg log_dir=log/avazu/w2_p1_16k dist.ddp -j 1x2 --script recsys/dlrm_main.py -- \
    --dataset_dir /data/scratch/RecSys/avazu_sample --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
    --profile_dir "tensorboard_log/avazu/w2_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 94458

# torchx run -s local_cwd -cfg log_dir=log/avazu/w4_p1_16k dist.ddp -j 1x4 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w4_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w8_p1_16k dist.ddp -j 1x8 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w8_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_32k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 32768 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_32k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_8k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 8192 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_8k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_4k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 4096 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_4k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_2k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 2048 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_2k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p1_1k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 1024 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p1_1k"  --buffer_size 0 --use_overlap --cache_sets 94458
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p10_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p10_16k"  --buffer_size 0 --use_overlap --cache_sets 944582
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p5_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p5_16k"  --buffer_size 0 --use_overlap --cache_sets 472291
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p2_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p2_16k"  --buffer_size 0 --use_overlap --cache_sets 188916
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p0_5_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p0_5_16k"  --buffer_size 0 --use_overlap --cache_sets 47229
# 
# torchx run -s local_cwd -cfg log_dir=log/avazu/w1_p0_1_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/avazu_sample --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#     --profile_dir "tensorboard_log/avazu/w1_p0_1_16k"  --buffer_size 0 --use_overlap --cache_sets 9445
# 
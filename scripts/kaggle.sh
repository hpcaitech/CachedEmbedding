#!/bin/bash

# For Colossalai enabled recsys
# criteo kaggle
# export DATAPATH=/data/scratch/RecSys/criteo_kaggle_data/
export DATAPATH=/data/criteo_kaggle_data/
export GPUNUM=4
export BATCHSIZE=16384
export CACHESIZE=337625
torchx run -s local_cwd -cfg log_dir=log/kaggle/w${GPUNUM}_p1_16k dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
    --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache --use_freq --use_lfu \
     --profile_dir "tensorboard_log/kaggle/w${GPUNUM}_p1_16k"  --buffer_size 0 --use_overlap --cache_sets ${CACHESIZE} 2>&1 | tee logs/colo_${GPUNUM}_${BATCHSIZE}_${CACHESIZE}.txt

# exit(0)

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w2_p1_16k dist.ddp -j 1x2 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w2_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w4_p1_16k dist.ddp -j 1x4 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 5e-1 --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w4_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w8_p1_16k dist.ddp -j 1x8 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 5e-1 --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w8_p1_16k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p1_32k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 32768 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p1_32k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p1_8k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 8192 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p1_8k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p1_4k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 4096 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p1_4k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p1_2k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 2048 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p1_2k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p1_1k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 1024 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p1_1k"  --buffer_size 0 --use_overlap --cache_sets 337625

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p10_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p10_16k"  --buffer_size 0 --use_overlap --cache_sets 3376257

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p5_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p5_16k"  --buffer_size 0 --use_overlap --cache_sets 1688128

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p2_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p2_16k"  --buffer_size 0 --use_overlap --cache_sets 675251

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p0_5_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p0_5_16k"  --buffer_size 0 --use_overlap --cache_sets 168812

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w1_p0_1_16k dist.ddp -j 1x1 --script recsys/dlrm_main.py -- \
#     --dataset_dir /data/criteo_kaggle_data --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size 16384 --use_sparse_embed_grad --use_cache --use_freq \
#      --profile_dir "tensorboard_log/kaggle/w1_p0_1_16k"  --buffer_size 0 --use_overlap --cache_sets 33762

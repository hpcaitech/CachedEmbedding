#!/bin/bash

# For Colossalai enabled recsys
# criteo kaggle
# export DATAPATH=/data/scratch/RecSys/criteo_kaggle_data/
export DATAPATH=/data/criteo_kaggle_data/
export GPUNUM=4
export BATCHSIZE=16384
export CACHESIZE=337625
# export USE_LFU=1
if [[ ${USE_LFU} == 1 ]];  then
LFU_FLAG="--use_lfu "
else
export LFU_FLAG=""
fi


torchx run -s local_cwd -cfg log_dir=log/kaggle/w${GPUNUM}_p1_16k dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
    --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache --use_freq ${LFU_FLAG} \
     --profile_dir "tensorboard_log/kaggle/w${GPUNUM}_p1_${BATCHSIZE}"  --buffer_size 0 --use_overlap --cache_sets ${CACHESIZE} 2>&1 | tee logs/colo_${GPUNUM}_${BATCHSIZE}_${CACHESIZE}_lfu_${USE_LFU}.txt

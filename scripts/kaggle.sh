#!/bin/bash

# For Colossalai enabled recsys
# criteo kaggle
export DATAPATH=/data/scratch/RecSys/criteo_kaggle_data/
# export DATAPATH=/data/criteo_kaggle_data/
export GPUNUM=1
export BATCHSIZE=16384
export CACHESIZE=337625
export USE_LFU=0
export USE_TABLE_SHARD=1
export EVAL_ACC=1

if [[ ${USE_LFU} == 1 ]];  then
LFU_FLAG="--use_lfu"
else
export LFU_FLAG=""
fi


if [[ ${USE_TABLE_SHARD} == 1 ]];  then
TABLE_SHARD_FLAG="--use_tablewise"
else
export TABLE_SHARD_FLAG=""
fi

if [[ ${EVAL_ACC} == 1 ]];  then
EVAL_ACC_FLAG="--eval_acc"
else
export EVAL_ACC_FLAG=""
fi



torchx run -s local_cwd -cfg log_dir=log/kaggle/w${GPUNUM}_p1_16k dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
    --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache ${LFU_FLAG} ${TABLE_SHARD_FLAG} ${EVAL_ACC_FLAG} \
    --profile_dir "tensorboard_log/kaggle/w${GPUNUM}_p1_${BATCHSIZE}"  --buffer_size 0 --use_overlap --cache_sets ${CACHESIZE} 2>&1 | tee logs/colo_${GPUNUM}_${BATCHSIZE}_${CACHESIZE}_lfu_${USE_LFU}_tw_${USE_TABLE_SHARD}.txt
# torchx run -s local_cwd -cfg log_dir=log/kaggle/w${GPUNUM}_p1_16k dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
#     --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache --use_freq --use_lfu \
#      --profile_dir "tensorboard_log/kaggle/w${GPUNUM}_p1_16k"  --buffer_size 0 --use_overlap --cache_sets ${CACHESIZE} 2>&1 | tee logs/colo_${GPUNUM}_${BATCHSIZE}_${CACHESIZE}.txt

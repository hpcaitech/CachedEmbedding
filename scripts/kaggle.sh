#!/bin/bash

# For Colossalai enabled recsys
# criteo kaggle
export DATAPATH=/data/scratch/RecSys/criteo_kaggle_data/
# export DATAPATH=/data/criteo_kaggle_data/
export GPUNUM=1
# export BATCHSIZE=16384
# export BATCHSIZE=1024
export CACHESIZE=337625
export CACHERATIO=0.01
export USE_LFU=1
export USE_TABLE_SHARD=0
export EVAL_ACC=0
export PREFETCH_NUM=2
export USE_ASYNC=0

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

if [[ ${USE_ASYNC} == 1 ]];  then
ASYNC_FLAG="--use_cache_mgr_async_copy"
else
export ASYNC_FLAG=""
fi

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}


mkdir -p colo_logs
for BATCHSIZE in 16384 #16384 #1024 2048 4096 16384
do
for PREFETCH_NUM in  1 #1 16
do
for USE_ASYNC in 0
do
set_n_least_used_CUDA_VISIBLE_DEVICES ${GPUNUM}

TASK_NAME="gpu_${GPUNUM}_bs_${BATCHSIZE}_pf_${PREFETCH_NUM}_cache_${CACHERATIO}_async_${USE_ASYNC}"
torchx run -s local_cwd -cfg log_dir=log/kaggle/${TASK_NAME} dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
    --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache --use_freq ${LFU_FLAG} ${TABLE_SHARD_FLAG} ${EVAL_ACC_FLAG} ${ASYNC_FLAG} \
    --profile_dir "tensorboard_log/kaggle/${TASK_NAME}"  --buffer_size 0 --use_overlap --cache_ratio ${CACHERATIO} --prefetch_num ${PREFETCH_NUM} 2>&1 | tee colo_logs/colo_${TASK_NAME}.txt
done
done
done

# torchx run -s local_cwd -cfg log_dir=log/kaggle/w${GPUNUM}_p1_16k dist.ddp -j 1x${GPUNUM} --script recsys/dlrm_main.py -- \
#     --dataset_dir ${DATAPATH} --pin_memory --shuffle_batches \
#     --learning_rate 1. --batch_size ${BATCHSIZE} --use_sparse_embed_grad --use_cache --use_freq --use_lfu \
#      --profile_dir "tensorboard_log/kaggle/w${GPUNUM}_p1_16k"  --buffer_size 0 --use_overlap --cache_sets ${CACHESIZE} 2>&1 | tee logs/colo_${GPUNUM}_${BATCHSIZE}_${CACHESIZE}.txt

#!/bin/bash
set -x

export DATASETPATH=/data/scratch/RecSys/criteo_preproc/
export PYTHONPATH=$HOME/codes/torchrec:$PYTHONPATH

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

export LOG_DIR="/data2/users/lcfjr/logs_1tb/a"
mkdir -p ${LOG_DIR}

export GPUNUM=2
export PREFETCH_NUM=8
export EVAL_ACC=0
export KERNELTYPE="colossalai" #"colossalai"
export SHARDTYPE="table"
export BATCHSIZE=1024
# export SHARDTYPE="uvm_lfu"
export EMB_DIM=128

if [[ ${EVAL_ACC} == 1 ]];  then
EVAL_ACC_FLAG="--eval_acc"
else
export EVAL_ACC_FLAG=""
fi


# For TorchRec baseline
# ${BATCHSIZE}*${GPUNUM} is the global batch size
for PREFETCH_NUM in 8 1
do
for BATCHSIZE in 8192 4096 2048 1024
do
for GPUNUM in 1 2 4
do
set_n_least_used_CUDA_VISIBLE_DEVICES ${GPUNUM}
export PLAN=k_${KERNELTYPE}_g_${GPUNUM}_bs_${BATCHSIZE}_sd_${SHARDTYPE}_pf_${PREFETCH_NUM}_eb_${EMB_DIM}

torchx run -s local_cwd -cfg log_dir=log/torchrec_terabyte/w1_16k dist.ddp -j 1x${GPUNUM} --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path ${DATASETPATH} --embedding_dim ${EMB_DIM} --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --shard_type ${SHARDTYPE} --kernel_type ${KERNELTYPE} --prefetch_num ${PREFETCH_NUM} ${EVAL_ACC_FLAG} \
    --profile_dir "" ${EVAL_ACC_FLAG} --limit_train_batches 10240000 2>&1 | tee ${LOG_DIR}/torchrec_${PLAN}.txt
done
done
done
# torchx run -s local_cwd -cfg log_dir=log/torchrec_terabyte/w2_16k dist.ddp -j 1x2 --script baselines/dlrm_main.py -- \
#     --in_memory_binary_criteo_path ${DATASETPATH} --embedding_dim 128 --pin_memory \
#     --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
#     --learning_rate 1. --batch_size 8192  --profile_dir "tensorboard_log/torchrec_terabyte/w2_16k"

# torchx run -s local_cwd -cfg log_dir=log/torchrec_terabyte/w4_16k dist.ddp -j 1x4 --script baselines/dlrm_main.py -- \
#     --in_memory_binary_criteo_path ${DATASETPATH} --embedding_dim 128 --pin_memory \
#     --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
#     --learning_rate 1. --batch_size 4096  --profile_dir "tensorboard_log/torchrec_terabyte/w4_16k"

# torchx run -s local_cwd -cfg log_dir=log/torchrec_terabyte/w8_16k dist.ddp -j 1x8 --script baselines/dlrm_main.py -- \
#     --in_memory_binary_criteo_path ${DATASETPATH} --embedding_dim 128 --pin_memory \
#     --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,128" --shuffle_batches \
#     --learning_rate 1. --batch_size 2048  --profile_dir "tensorboard_log/torchrec_terabyte/w8_16k"

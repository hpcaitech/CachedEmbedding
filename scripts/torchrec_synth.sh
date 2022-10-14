#!/bin/bash
set -x

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

export DATAPATH=/data/scratch/RecSys/embedding_bag
# export DATAPATH=custom
export SCALE="512M" # "4M ""52M" "512M"
export EVAL_ACC=0
export EMB_DIM=128

if [[ ${EVAL_ACC} == 1 ]];  then
EVAL_ACC_FLAG="--eval_acc"
else
export EVAL_ACC_FLAG=""
fi

# local batch size
# 4
# export BATCHSIZE=1024
# export BATCHSIZE=1024

# export BATCHSIZE=4096
# 2
# export BATCHSIZE=8192
# 1

mkdir -p logs
for PREFETCH_NUM in 1 32 4 8 16 #8 16 32
do
for GPUNUM in 1 # 1 # 2
do
for BATCHSIZE in 256 #2048 4096 1024 #8192 512 ##16384 8192 4096 2048 1024 512     
do
for SHARDTYPE in  "table"
do
for KERNELTYPE in "colossalai" # "uvm_lfu" # "colossalai" # "uvm_lfu" # "colossalai"
do
# For TorchRec baseline
set_n_least_used_CUDA_VISIBLE_DEVICES ${GPUNUM}
export PLAN=g${GPUNUM}_bs_${BATCHSIZE}_${SHARDTYPE}_pf_${PREFETCH_NUM}_eb_${EMB_DIM}
rm -rf ./tensorboard_log/torchrec_synth/
# env CUDA_LAUNCH_BLOCKING=1 
# timeout -s SIGKILL 30m 
torchx run -s local_cwd -cfg log_dir=log/torchrec_synth/${PLAN} dist.ddp -j 1x${GPUNUM} --script baselines/dlrm_main.py -- \
    --in_memory_binary_criteo_path ${DATAPATH} --kaggle --embedding_dim ${EMB_DIM} --pin_memory \
    --over_arch_layer_sizes "1024,1024,512,256,1" --dense_arch_layer_sizes "512,256,${EMB_DIM}" --shuffle_batches \
    --learning_rate 1. --batch_size ${BATCHSIZE} --profile_dir "" --shard_type ${SHARDTYPE} --kernel_type ${KERNELTYPE}  \
    --synth_size ${SCALE} \
    --synth_size ${SCALE} --prefetch_num ${PREFETCH_NUM} ${EVAL_ACC_FLAG} 2>&1 | tee logs/torchrec_${PLAN}.txt
done
done
done
done
done

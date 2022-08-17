#!/bin/bash
# Due to the distributed GPU settings customized by NVTabular,
# we need some awkward wrapper to initialize the distributed settings in PyTorch code.
#
# Basically, we need to assign each process with a single & different device id to enable NVTabular,
# To cope with the visible device by pytorch,
# we force the OMPI_COMM_WORLD_LOCAL_RANK to 0 (Please refer to dlrm_main.py),
# and thus set each visible device in each process to cuda:0 (this part is done by colossalai under the hood).

# Usage:
#   mpirun -np <num_proc> bash dist_wrapper.sh python <training script> [training args]
#   torchrun --nnode=1 --nproc_per_node=<num_proc> --no_python bash dist_wrapper.sh python <training script> \
#     [training args]
#
# hovorodrun might also work since it invokes mpirun.

# Get local process ID from OpenMPI or Slurm
if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
    LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
elif [ -n "${SLURM_LOCALID:-}" ]; then
    LOCAL_RANK="${SLURM_LOCALID}"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
else
    device_list=(${CUDA_VISIBLE_DEVICES//","/ })
    export CUDA_VISIBLE_DEVICES=${device_list[$LOCAL_RANK]}
fi
export NVT_TAG=1
exec "$@"

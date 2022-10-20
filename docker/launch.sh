DATASET_PATH=/data/scratch/RecSys

docker run --rm -it -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e PYTHONPATH=/workspace/code -v `pwd`:/workspace/code -v ${DATASET_PATH}:/data -w /workspace/code --ipc=host hpcaitech/cacheembedding:0.1.4 /bin/bash

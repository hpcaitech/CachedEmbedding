# FreqCacheEmbedding

This repo contains the implementation of FreqCacheEmbedding, which extends the vanilla
[PyTorch EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) 
with cache mechanism to enable heterogeneous training for large scale recommendation models.

### Dataset  
1. [Criteo Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data)
2. [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)

Basically, the preprocessing processes are derived from 
[Torchrec's utilities](https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/scripts/npy_preproc_criteo.py) 
and [Avazu kaggle community](https://www.kaggle.com/code/leejunseok97/deepfm-deepctr-torch)
Please refer to `recsys/datasets/preprocess_scripts` dir to see the details.

During the time this repo was built, another commonly adopted dataset, 
[Criteo 1TB](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) 
is unavailable (see this [issue](https://github.com/pytorch/torchrec/issues/245)).
We will append its preprocessing & running scripts very soon.

### Command  
All the commands to run the FreqCacheEmbedding enabled recommendations models are presented in `run.sh`
 
### Model  
Currently, this repo only contains DLRM & DeepFM models, 
and we are working on testing more recommendation models.
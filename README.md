# FreqCacheEmbedding

This repo contains the implementation of FreqCacheEmbedding, which extends the vanilla
[PyTorch EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag) 
with **Freqency-Aware Cache Embedding** from [ColossalAI](https://github.com/hpcaitech/ColossalAI) to enable heterogeneous training for large scale recommendation models.

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

### Performance

The DLRM performance on three datasets using ColossalAI version (this repo) and torchrec (with UVM) is shown as follows. The cache ratio of FreqAwareEmbedding is set as 1%.

|            |   method   | AUROC over Test after 1 Epoch | Acc over test | Throughput | Time to Train 1 Epoch | GPU memory allocated (GB) | GPU memory reserved (GB) | CPU memory usage (GB) |
|:----------:|:----------:|:-----------------------------:|:-------------:|:----------:|:---------------------:|:-------------------------:|:------------------------:|:---------------------:|
| criteo 1TB | ColossalAI |          0.791299403          |  0.967155457  |   42 it/s  |         1h40m         |            3.75           |           5.04           |         94.39         |
|            |  torchrec  |           0.79515636          |  0.967177451  |   45 it/s  |         1h25m         |           66.54           |           68.43          |          7.7          |
|   kaggle   | ColossalAI |          0.776755869          |  0.779025435  |   50 it/s  |          49s          |            0.9            |           2.14           |         34.66         |
|            |  torchrec  |          0.786652029          |  0.782288849  |   81 it/s  |          30s          |           16.13           |           17.99          |         13.89         |
|   avazue   | ColossalAI |          0.701478183          |  0.821486056  |   72 it/s  |          31s          |            0.31           |           1.06           |         16.89         |
|            |  torchrec  |          0.725972056          |  0.824484706  |  111 it/s  |          21s          |            4.53           |           5.83           |         12.25         |

### Cite us
```
@article{fang2022frequency,
  title={A Frequency-aware Software Cache for Large Recommendation System Embeddings},
  author={Fang, Jiarui and Zhang, Geng and Han, Jiatong and Li, Shenggui and Bian, Zhengda and Li, Yongbin and Liu, Jin and You, Yang},
  journal={arXiv preprint arXiv:2208.05321},
  year={2022}
}
```

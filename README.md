# Exploration of TorchRec

This repo is derived from the official implementation of 
[DLRM](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm).

However, we make several modifications.
### Diff
1. Dataset  
During the time this repo is built, the [Criteo 1TB](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) 
is unavailable (see this [issue](https://github.com/pytorch/torchrec/issues/245)).
So, I adapted the preprocessing steps so that the [Criteo kaggle](https://ailab.criteo.com/ressources/) dataset can be loaded.  
Please refer to `criteo_kaggle` dir to see the details.
   

2. Command  
I added an option `--kaggle` in the ArgumentParser and altered a few lines of code, so that the model would construct corresponding embedding tables for 
   the sparse features in this dataset, and the 7-day dataset can be correctly split into train/val/test parts.
   

3. Model  
Currently, this repo only contains DLRM. 
   Actually, the model can be directly imported from torchrec.models.dlrm.  
   I copied it into `models` dir, because there may be some future work to be done with the model.
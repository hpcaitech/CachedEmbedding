# Dataset preprocess
## Criteo Kaggle
1. download data from: https://ailab.criteo.com/ressources/
2. convert tsvs to npy files
```bash
python npy_preproc_criteo.py --input_dir <kaggle_tsv_dir> --output_dir <kaggle_npy_dir>
```
3. split to train/val/test files
```bash
python split_criteo_kaggle.py
```
You might need to change the `'/data/scratch/RecSys/criteo_kaggle_npy'` to `<kaggle_npy_dir>` in this file

## Avazu
1. download data from: https://www.kaggle.com/c/avazu-ctr-prediction/data
2. convert tsv files to npy files:
```bash
python npy_preproc_avazu.py --input_dir <avazu_dir> --output_dir <avazu_npy_dir>
```
3. split train/val/test files
```bash
python npy_preproc_avazu.py --input_dir <avazu_npy_dir> --output_dir <avazu_split_dir> --is_split
```

## Criteo Terabyte
1. download tsv source file from: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/  
   Note that this requires around 1TB disk space.
2. clone the torchrec repo:
```bash
git clone https://github.com/pytorch/torchrec.git
cd torchrec/torchrec/datasets/scripts/nvt/
```
3. conduct the first two python script
```bash
python convert_tsv_to_parquet.py -i <terabyte_tsv_dir> -o <terabyte_parquet_dir>
python process_criteo_parquet.py -b <terabyte_parquet_dir> -s
```
You might need to use the dockerfile to install nvtabular,
since its installation requires a CUDA version different from our experiment setup
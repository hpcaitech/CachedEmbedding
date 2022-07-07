colossalai run --nproc_per_node 2 --master_port 29501 deepfm_main.py --config deepfm_config.py --dataset 'criteo' --enable_qr  --use_clip --repeated_run 1 --group 'deepfm-dbstream'

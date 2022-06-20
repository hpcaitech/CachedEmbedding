colossalai run --nproc_per_node 2 --master_port 29501 main.py --config config.py --dataset 'criteo' --enable_qr  --use_clip --use_wandb --repeated_run 1 --group 'deepfm-dbstream'

colossalai run --nproc_per_node 4 --master_port 29501 deepfm_main.py --config deepfm_config.py --dataset 'criteo' --repeated_run 1 --group 'deepfm-dbstream'

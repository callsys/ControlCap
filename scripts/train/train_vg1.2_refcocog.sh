#! /bin/bash

ls

# train controlcap on vg1.2 + refcocog
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/union/vg_refcocog_5e.yaml
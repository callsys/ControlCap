#! /bin/bash

ls

# train controlcap on vg1.2
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/vg/vg1.2_5e.yaml
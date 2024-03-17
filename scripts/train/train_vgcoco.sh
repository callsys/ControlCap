#! /bin/bash

ls

# train controlcap on vgcoco
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/vg/vgcoco_5e.yaml
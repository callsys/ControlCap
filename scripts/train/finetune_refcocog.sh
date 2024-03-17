#! /bin/bash

ls

ckpt=$1
echo $ckpt

# eval controlcap on refcocog
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/refcoco/refcocog_ft.yaml --options run.load_ckpt_path=$ckpt
# Metrics ({'METEOR': 17.807078240962984, 'CIDEr': 112.80247003383761})

#! /bin/bash

ls

ckpt=$1
echo $ckpt

# eval controlcap on vg_reg
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg/vg_reg.yaml --options run.load_ckpt_path=$ckpt

#! /bin/bash

ls

ckpt=$1
echo $ckpt

# eval controlcap on refcocog
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/refcoco/refcocog_reg.yaml --options run.load_ckpt_path=$ckpt

#! /bin/bash

ls

ckpt="/Workspace/ZhaoYuzhong/ControlCap/output/train/vg_refcocog_5e/20240316102/checkpoint_4.pth"
echo $ckpt

# eval controlcap on vg1.0
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg/vg1.0_densecap.yaml --options run.load_ckpt_path=$ckpt
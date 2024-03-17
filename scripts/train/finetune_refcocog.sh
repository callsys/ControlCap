#! /bin/bash

ls

ckpt="/Workspace/ZhaoYuzhong/ControlCap/output/train/refcocog_ft/20240317123/checkpoint_2.pth"
echo $ckpt

# eval controlcap on refcocog
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/refcoco/refcocog_ft.yaml --options run.load_ckpt_path=$ckpt
# Metrics ({'METEOR': 17.807078240962984, 'CIDEr': 112.80247003383761})
#! /bin/bash

ls

# train reem on vg
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/vg/vg1.0.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/vg/vg1.2.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/train/vg/vgcoco.yaml

# train rrecog on vg and extra data



# evaluate rrecog on vg with gt bboxes
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vg1.0.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vg1.2.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vgcoco.yaml

# evaluate rrecog on vg with grit predict bboxes
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vg1.0_grit.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vg1.2_grit.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/eval/vg_rrecog_t5/vgcoco_grit.yaml



# evaluate rrecog with given result file
python eval.py --cfg-path configs/eval/vg_rrecog_t5/vgcoco.yaml --res-path data/vg/annotations/rrecog/vgcoco/test_grit_vgcoco.json --metric

# visualize rrecog with given result file
python eval.py --cfg-path configs/eval/vg_rrecog_t5/vgcoco.yaml --res-path data/vg/annotations/rrecog/vgcoco/test_grit_vgcoco.json --viz



# annotate other datasets with conditional text generation
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/annotate_synth_data/coco_rrecog_t5.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/annotate_synth_data/paco_rrecog_t5.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/annotate_synth_data/part_imagenet_rrecog_t5.yaml
python -m torch.distributed.run --nproc_per_node=8 --master_port=29600 train.py --cfg-path configs/annotate_synth_data/pascal_part_rrecog_t5.yaml
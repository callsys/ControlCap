#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./

CUDA_VISIBLE_DEVICES=0 python rrecog/commom/sam_preprocess.py --gpu 0 &
CUDA_VISIBLE_DEVICES=1 python rrecog/commom/sam_preprocess.py --gpu 1 &
CUDA_VISIBLE_DEVICES=2 python rrecog/commom/sam_preprocess.py --gpu 2 &
CUDA_VISIBLE_DEVICES=3 python rrecog/commom/sam_preprocess.py --gpu 3 &
CUDA_VISIBLE_DEVICES=4 python rrecog/commom/sam_preprocess.py --gpu 4 &
CUDA_VISIBLE_DEVICES=5 python rrecog/commom/sam_preprocess.py --gpu 5 &
CUDA_VISIBLE_DEVICES=6 python rrecog/commom/sam_preprocess.py --gpu 6 &
CUDA_VISIBLE_DEVICES=7 python rrecog/commom/sam_preprocess.py --gpu 7

python rrecog/commom/sam_preprocess.py --collect
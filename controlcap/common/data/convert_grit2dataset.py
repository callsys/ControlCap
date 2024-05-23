import os
# import json
import ujson as json
import tqdm
import time
import string
import itertools
import cv2
import copy
import numpy as np
import concurrent.futures as futures
from collections import Counter
import matplotlib.pyplot as plt
import selectivesearch
import argparse
from pycocotools.coco import COCO

from controlcap.common.data.processors import BaseProcessor
save_path = "data/grit/controlcap/train_fused_4.8m.json"
grit_fuse = "data/grit/controlcap/tmp/train_fused_4.8m.json"

file = json.load(open(grit_fuse, "r"))

print(file.keys())
print(file["dataset"])
file["dataset"]["image_root"] = "/home/ZhaoYuzhong/Dataset/lavis/grit/images/"
print(file["dataset"])
print(file["images"][0])

for img in tqdm.tqdm(file["images"]):
    img["extra_info"] = dict()

for ann in tqdm.tqdm(file["annotations"]):
    bbox = ann["bbox"]
    x1, y1, x2, y2 = bbox
    seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]
    ann["segmentation"] = seg
    ann.pop("bbox")
    parser_result = ann["extra_info"].get("parser_result", dict())
    parser_result.pop("graph")
    new_extra_info = {"parser_result": parser_result}
    ann.pop("extra_info")
    ann["extra_info"] = new_extra_info
print(file["annotations"][0])

with open(save_path, "w") as fw:
    json.dump(file, fw)

print(0)

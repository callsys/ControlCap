import os
import json
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

class SYNTHProcessor(BaseProcessor):
    def __init__(self,ann_root = ""):
        super().__init__()
        self.version = "synth"
        self.ann_root = ann_root

        save_files = [ann_root.replace(".json", "_synth.json")]

        self.save_files = save_files
        self.save_file = ""

    def parse_anns(self):
        dataset_sample = json.load(open(self.ann_root, "r"))

        for ann in tqdm.tqdm(dataset_sample["annotations"]):
            pred_result = ann["extra_info"].get("pred_result", None)
            if pred_result is not None:
                caption = pred_result["caption"]
                ann["caption"] = caption

        non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
        images = dataset_sample["images"]
        filter_images = [image for image in images if image["id"] in non_empty_image_ids]
        dataset_sample["images"] = filter_images

        print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
        print(f"[INFO]:\t{len(dataset_sample['images'])} images")
        print(f"[INFO]:\t({self.version}) dump annotation to ({self.save_file})")
        with open(self.save_file, "w") as fw:
            json.dump(dataset_sample, fw)

        return self.save_file

    def process(self):
        for save_file in self.save_files:
            self.save_file = save_file
            save_file = self.parse_anns()
            save_file = self.parse_graph(save_file)
            save_file = self.parse_tags(save_file)





def parse_args():
    parser = argparse.ArgumentParser(description="Convert ControlCap generated captions to dataset")
    parser.add_argument("--ann-root", default="data/coco2017/controlcap/coco_train.json", help="path to annotation root.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    processor = SYNTHProcessor(ann_root=args.ann_root)
    processor.process()

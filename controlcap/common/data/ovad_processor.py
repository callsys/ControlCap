import os
import json

import sng_parser
import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
from pycocotools.coco import COCO

from processors import BaseProcessor


class OVADProcessor(BaseProcessor):
    def __init__(self,
                 version = "refcoco",
                 image_root = "",
                 ann_root = "",
                 save_dir = ""):
        super().__init__()

        self.version = version
        self.image_root = image_root
        self.ann_root = ann_root
        self.save_dir = save_dir

        if not os.path.exists(os.path.join(save_dir)):
            os.system(f"mkdir -p {save_dir}")

        ann_roots = [ann_root]
        save_files = [os.path.join(save_dir, "test_ovad.json")]

        self.ann_roots = ann_roots
        self.save_files = save_files
        self.ann_root = ""
        self.save_file = ""



    def parse_anns(self):
        dataset_sample = {"dataset": {"description": self.version,
                                      "image_root": self.image_root,
                                      "extra_info": dict()},
                          "images": [],
                          "annotations": []}

        gt = COCO(self.ann_root)
        imgs = gt.imgs

        dataset_sample["attrs"] = gt.dataset["attributes"]
        dataset_sample["cats"] = gt.dataset["categories"]

        for img_id, img in tqdm.tqdm(imgs.items()):
            image = img["file_name"]
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            height, width = img["height"], img["width"]
            image_sample = {"id": img_id,
                            "file_name": image,
                            "height": height,
                            "width": width,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            anns = gt.imgToAnns[img_id]
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                bbox = [x, y, x+w, y+h]
                
                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann_sample = {"id": ann["id"],
                              "image_id": img_id,
                              "caption": gt.cats[ann["category_id"]]["name"],
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": {"ovad_result":{"att_vec": ann["att_vec"]}}}
                dataset_sample["annotations"].append(ann_sample)

        non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
        images = dataset_sample["images"]
        filter_images = [image for image in images if image["id"] in non_empty_image_ids]
        dataset_sample["images"] = filter_images

        print(f"[INFO]:\tdataset ({self.version})")
        print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
        print(f"[INFO]:\t{len(dataset_sample['images'])} images")
        print(f"[INFO]:\tdump annotation to ({self.save_file})")
        with open(self.save_file, "w") as fw:
            json.dump(dataset_sample, fw)
        return self.save_file

    def process(self):
        for ann_root, save_file in zip(self.ann_roots, self.save_files):
            self.ann_root = ann_root
            self.save_file = save_file
            save_file = self.parse_anns()
            # save_file = self.parse_graph(save_file)
            # save_file = self.parse_tags(save_file)


def parse_args():
    parser = argparse.ArgumentParser(description="OVAD data processor")
    parser.add_argument("--version", choices=["ovad"], default="ovad", help="dataset version.")
    parser.add_argument("--image-root", default="data/coco2017/images/images", help="path to image root.")
    parser.add_argument("--ann-root", default="data/coco2017/ovad/ovad2000.json", help="path to annotation root.")
    parser.add_argument("--save-dir", default="data/coco2017/controlcap/", help="path to save dir.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    processor = OVADProcessor(version=args.version,
                            image_root=args.image_root,
                            ann_root=args.ann_root,
                            save_dir=args.save_dir)
    processor.process()
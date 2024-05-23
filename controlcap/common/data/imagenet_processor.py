import os
import json
import pickle
import sng_parser
import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from nltk.corpus import wordnet
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from processors import BaseProcessor


class ImageNetProcessor(BaseProcessor):
    def __init__(self,
                 version = "imagenet",
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

        if self.version == "imagenet-s":
            splits = ["test"]
            save_files = []
            ann_roots = []
            for split in splits:
                save_files.append(os.path.join(save_dir, f"{split}_imagenet_s.json"))
                ann_roots.append(os.path.join(ann_root, f"imagenet_919.json"))
        else:
            save_files = []
            ann_roots = []

        self.ann_roots = ann_roots
        self.save_files = save_files
        self.ann_root = ""
        self.save_file = ""

    def parse_anns_imagenet_s(self):
        dataset_sample = {"dataset": {"description": self.version,
                                      "image_root": self.image_root,
                                      "extra_info": dict()},
                          "images": [],
                          "annotations": []}

        gt = json.load(open(self.ann_root, 'r'))

        cats = []
        for img in gt:
            if img['category_word'] not in cats:
                cats.append(img['category_word'])
            img['category_id'] = len(cats) - 1
        classes = []
        for cat_word in cats:
            synset = wordnet.synset_from_pos_and_offset('n', int(cat_word[1:]))
            synonyms = [x.name() for x in synset.lemmas()]
            classes.append(synonyms[0])

        for img_id, img in enumerate(tqdm.tqdm(gt)):
            image = img["image_pth"]
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            height, width = [int(el) for el in img["mask"]["size"]]
            image_sample = {"id": img_id,
                            "file_name": image,
                            "height": height,
                            "width": width,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            mask = maskUtils.decode(img['mask'])

            try:
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
                bbox = [int(el) for el in bbox]
                seg = img["mask"]
            except:
                x1, y1, x2, y2 = 0, 0, width, height
                bbox = [x1, y1, x2, y2]
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

            synset = wordnet.synset_from_pos_and_offset('n', int(img["category_word"][1:]))
            synonyms = [x.name() for x in synset.lemmas()]
            caption = synonyms[0]

            ann_sample = {"id": img_id,
                          "image_id": img_id,
                          "caption": caption,
                          "bbox": bbox,
                          "segmentation": seg,
                          "extra_info": {"imagenet_s_result": {"category_word": img["category_word"],
                                                               "category_id": img["category_id"]}}}
            dataset_sample["annotations"].append(ann_sample)

        non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
        images = dataset_sample["images"]
        filter_images = [image for image in images if image["id"] in non_empty_image_ids]
        dataset_sample["images"] = filter_images

        self.save_file_class = self.save_file.replace(".json", "_class.json")
        print(f"[INFO]:\tdump class annotation to ({self.save_file_class})")
        with open(self.save_file_class, "w") as fw:
            json.dump(classes, fw)

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
            if "imagenet-s" in self.version:
                save_file = self.parse_anns_imagenet_s()
            # save_file = self.parse_graph(save_file)
            # save_file = self.parse_tags(save_file)


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet data processor")
    parser.add_argument("--version", choices=["imagenet-s"], default="imagenet-s", help="dataset version.")
    parser.add_argument("--image-root", default="data/imagenet/images/", help="path to image root.")
    parser.add_argument("--ann-root", default="data/imagenet/annotations/", help="path to annotation root.")
    parser.add_argument("--save-dir", default="data/imagenet/controlcap/", help="path to save dir.")
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
    processor = ImageNetProcessor(version=args.version,
                                  image_root=args.image_root,
                                  ann_root=args.ann_root,
                                  save_dir=args.save_dir)
    processor.process()
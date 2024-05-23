import os
import json
import pickle
import sng_parser
import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
from pycocotools.coco import COCO

from processors import BaseProcessor


class COCOProcessor(BaseProcessor):
    def __init__(self,
                 version = "coco",
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

        if self.version == "coco":
            splits = ["train", "val"]
            save_files = []
            ann_roots = []
            for split in splits:
                save_files.append(os.path.join(save_dir, f"{split}.json"))
                ann_roots.append(os.path.join(ann_root, f"instances_{split}2017.json"))
        elif self.version == "coco-caption":
            save_files = [os.path.join(save_dir, f"train_cap.json")]
            ann_roots = [os.path.join(ann_root, "captions_train2017.json")]
        elif self.version == "lvis":
            splits = ["train", "val"]
            save_files = []
            ann_roots = []
            for split in splits:
                save_files.append(os.path.join(save_dir, f"{split}_lvis.json"))
                ann_roots.append(os.path.join(ann_root, f"lvis_v1_{split}.json"))
        elif self.version == "lvis-yolow":
            save_files = []
            ann_roots = []
            save_files.append(os.path.join(save_dir, f"test_lvis_yolow.json"))
            ann_roots.append(os.path.join(ann_root, "yolow.pkl"))
        else:
            save_files = []
            ann_roots = []

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
                seg = ann["segmentation"]

                ann_sample = {"id": ann["id"],
                              "image_id": img_id,
                              "caption": gt.cats[ann["category_id"]]["name"],
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": {"coco_result": {"category": gt.cats[ann["category_id"]]["name"],
                                                             "category_id": ann["category_id"]}}}
                dataset_sample["annotations"].append(ann_sample)

        non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
        images = dataset_sample["images"]
        filter_images = [image for image in images if image["id"] in non_empty_image_ids]
        dataset_sample["images"] = filter_images

        classes = [v["name"] for k, v in gt.cats.items()]
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

    def parse_anns_cap(self):
        dataset_sample = {"dataset": {"description": self.version,
                                      "image_root": self.image_root,
                                      "extra_info": dict()},
                          "images": [],
                          "annotations": []}

        gt = COCO(self.ann_root)
        imgs = gt.imgs

        for img_id, img in tqdm.tqdm(imgs.items()):
            image = img["file_name"]
            rename_image = image.split("_")[-1]
            abs_image_path = os.path.join(self.image_root, rename_image)
            assert os.path.exists(abs_image_path)
            height, width = img["height"], img["width"]
            image_sample = {"id": img_id,
                            "file_name": rename_image,
                            "height": height,
                            "width": width,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            anns = gt.imgToAnns[img_id]
            for ann in anns:
                bbox = [0, 0, width, height]

                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann_sample = {"id": ann["id"],
                              "image_id": img_id,
                              "caption": ann["caption"].strip(". ").lower(),
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": dict()}
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

    def parse_anns_lvis(self):
        dataset_sample = {"dataset": {"description": self.version,
                                      "image_root": self.image_root,
                                      "extra_info": dict()},
                          "images": [],
                          "annotations": []}

        def cat2cap(cat):
            cap = cat
            if "_" in cap:
                cap = cap.replace("_", " ")



        gt = COCO(self.ann_root)
        imgs = gt.imgs

        dataset_sample["cats"] = gt.dataset["categories"]

        for img_id, img in tqdm.tqdm(imgs.items()):
            image = os.path.basename(img["coco_url"])
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            height, width = img["height"], img["width"]
            image_sample = {"id": img_id,
                            "file_name": image,
                            "height": height,
                            "width": width,
                            "extra_info": {"coco_url": img["coco_url"],
                                           "neg_category_ids": img["neg_category_ids"],
                                           "not_exhaustive_category_ids": img["not_exhaustive_category_ids"]}}
            dataset_sample["images"].append(image_sample)

            anns = gt.imgToAnns[img_id]
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                bbox = [x, y, x+w, y+h]
                seg = ann["segmentation"]

                ann_sample = {"id": ann["id"],
                              "image_id": img_id,
                              "caption": gt.cats[ann["category_id"]]["def"],
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": {"lvis_result": {"category": gt.cats[ann["category_id"]]["name"],
                                                             "category_id": ann["category_id"]}}}
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

    def parse_anns_lvis_yolow(self):
        dataset_sample = {"dataset": {"description": self.version,
                                      "image_root": self.image_root,
                                      "extra_info": dict()},
                          "images": [],
                          "annotations": []}

        preds = pickle.load(open(self.ann_root, "rb"))

        ann_id = 0
        for pred in tqdm.tqdm(preds):
            image = os.path.basename(pred["img_path"])
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            height, width = pred["ori_shape"]
            img_id = pred["img_id"]
            image_sample = {"id": img_id,
                            "file_name": image,
                            "height": height,
                            "width": width,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            labels = pred["pred_instances"]["labels"]
            scores = pred["pred_instances"]["scores"]
            bboxes = pred["pred_instances"]["bboxes"]

            for label, score, bbox in zip(labels, scores, bboxes):
                bbox = bbox.tolist()
                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann_sample = {"id": ann_id,
                              "image_id": img_id,
                              "caption": "",
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": {"yolow_result": {"label": int(label),
                                                              "score": float(score)}}}
                ann_id = ann_id + 1
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
            if "caption" in self.version:
                save_file = self.parse_anns_cap()
                save_file = self.parse_graph(save_file)
                save_file = self.parse_tags(save_file)
            elif "lvis" == self.version:
                save_file = self.parse_anns_lvis()
                save_file = self.parse_graph(save_file)
                save_file = self.parse_tags(save_file)
            elif "lvis-yolow" == self.version:
                save_file = self.parse_anns_lvis_yolow()
            else:
                save_file = self.parse_anns()
            # save_file = self.parse_graph(save_file)
            # save_file = self.parse_tags(save_file)


def parse_args():
    parser = argparse.ArgumentParser(description="COCO data processor")
    parser.add_argument("--version", choices=["coco", "coco-caption", "lvis", "lvis-minival", "lvis-yolow"], default="coco", help="dataset version.")
    parser.add_argument("--image-root", default="data/coco2017/images/images", help="path to image root.")
    parser.add_argument("--ann-root", default="data/coco2017/annotations/", help="path to annotation root.")
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
    processor = COCOProcessor(version=args.version,
                            image_root=args.image_root,
                            ann_root=args.ann_root,
                            save_dir=args.save_dir)
    processor.process()
import os
import cv2
import copy
import random
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from lavis.datasets.datasets.base_dataset import BaseDataset

class ControlCapDataset(BaseDataset):
    def __init__(self,
                 vis_processor,
                 text_processor,
                 split,
                 **kwargs):
        self.split = split
        self.with_seg = kwargs.get("with_seg", False)
        self.obj_lvl = kwargs.get("obj_lvl", True)
        self.max_imgs = int(kwargs.get("max_imgs", int(1e7)))
        self.max_objs = int(kwargs.get("max_objs", int(1e7)))
        self.ann_files = kwargs.get("annotations", None)
        tag_list = kwargs.get("tag_list", "controlcap/common/tagging/ram_tag_list.txt")
        with open(tag_list, "r") as fr:
            self.tag_list = fr.readlines()
        self.tag_list = [tag.strip() for tag in self.tag_list]
        self.num_tags = len(self.tag_list)

        ann_files = self.ann_files[split]
        if self.split == "train":
            self.format_ann_files_train(ann_files)
        else:
            self.format_ann_files_eval(ann_files)

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        if self.vis_processor is not None:
            self.vis_processor.with_seg = self.with_seg

    def format_ann_files_train(self, ann_files):
        self.annotation = []
        for ann_file in ann_files:
            file = COCO(ann_file)
            self.dataset_info = file.dataset.get("dataset", dict())
            for img in list(file.imgs.values())[:self.max_imgs]:
                ann = dict()
                ann.update(self.dataset_info)
                ann.update(img)
                objs = file.imgToAnns[img["id"]]
                train_objs = []
                for obj in objs:
                    train_objs.append(obj)
                if len(train_objs) == 0:
                    continue
                if self.obj_lvl:
                    for train_obj in train_objs:
                        obj_ann = copy.deepcopy(ann)
                        obj_ann["objs"] = [train_obj]
                        self.annotation.append(obj_ann)
                else:
                    ann["objs"] = train_objs
                    self.annotation.append(ann)

    def format_ann_files_eval(self, ann_files):
        assert len(ann_files)==1
        ann_file = ann_files[0]
        self.annotation = []
        file = COCO(ann_file)
        self.dataset_info = file.dataset.get("dataset", dict())
        for img in list(file.imgs.values())[:self.max_imgs]:
            ann = dict()
            ann.update(self.dataset_info)
            ann.update(img)
            objs = file.imgToAnns[img["id"]]
            eval_objs = []
            for obj in objs:
                eval_objs.append(obj)
            ann["objs"] = eval_objs
            self.annotation.append(ann)

    def visualize(self, ann):
        dataset_root = ann["dataset_root"]
        image_root = ann["image_root"]
        img_path = os.path.join(dataset_root, image_root, ann["file_name"])
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        expand_ratio = 5
        captions_to_draw = []
        captions = []
        for obj in ann["objs"]:
            for cap_type, cap in obj["caption"].items():
                captions.extend(cap)
            caption = " | ".join(captions)
            seg = obj["segmentation"]
            if isinstance(seg, list):
                mask = np.zeros((h, w), np.uint8)
                for seg_ in seg:
                    mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
            else:
                mask = mask_util.frPyObjects(seg, *seg["size"])
                mask = mask_util.decode(mask)
            pos = (np.array(np.nonzero(mask)).min(1).astype(np.int64)[1] * expand_ratio,
                   np.array(np.nonzero(mask)).mean(1).astype(np.int64)[0] * expand_ratio)

            mask = mask.astype(np.bool_)
            mask_3d_color = np.zeros((h, w, 3), dtype="uint8")
            rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
            mask_3d_color[mask] = rgb
            image[mask] = image[mask] * 0.5 + mask_3d_color[mask] * 0.5

            captions_to_draw.append((caption, pos))

        dsize = (w * expand_ratio, h * expand_ratio)
        image = cv2.resize(image, dsize)

        for caption, pos in captions_to_draw:
            cv2.putText(image, caption, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 255, 255),
                            thickness=2)

        return image

    def get_objs(self, ann):
        objs = ann["objs"]
        if self.split == "train":
            # to avoid oom
            if len(objs) > self.max_objs:
                selects = np.arange(0, len(objs))
                np.random.shuffle(selects)
                selects = selects[:self.max_objs]
                objs = [objs[idx] for idx in selects]
        return objs

    def get_vision_data(self, ann, objs):
        image_path = os.path.join(ann["image_root"], ann["file_name"])
        image = Image.open(image_path).convert("RGB")
        segs = [obj["segmentation"] for obj in objs]
        data = self.vis_processor(image, segs)
        return data

    def get_language_data(self, ann, objs):
        if self.split == "train":
            caps = [obj["caption"] for obj in objs]
            tags = torch.zeros([len(objs), self.num_tags * 2])
            for i, obj in enumerate(objs):
                extra_info = obj["extra_info"]
                if "tag_set1" in extra_info.get("parser_result", dict()):
                    tag_set1 = extra_info["parser_result"].get("tag_set1", [])
                    for j in tag_set1:
                        tags[i, j] = 1
                if "tag_set2" in extra_info.get("parser_result", dict()):
                    tag_set2 = extra_info["parser_result"].get("tag_set2", [])
                    for j in tag_set2:
                        tags[i, j + self.num_tags] = 1
        else:
            caps = [obj["caption"] for obj in objs]
            tags = torch.zeros([len(objs), self.num_tags * 2])

        caps = [self.text_processor(cap) for cap in caps]

        return {"caps": caps, "tags": tags.to(torch.long)}

    def __getitem__(self, index):
        try:
            ann = self.annotation[index]
            objs = self.get_objs(ann)
            vision_data = self.get_vision_data(ann, objs)
            language_data = self.get_language_data(ann, objs)

            return {
                "image": vision_data["image"],
                "region_images": vision_data["region_images"],
                "bboxes": vision_data["bboxes"],
                "caps": language_data["caps"],
                "tags": language_data["tags"],
                "ids": [obj["id"] for obj in objs]}
        except:
            print(f"find an invalid sample [{str(ann)}]")
            return self.__getitem__(index + 1)

    def collater(self, samples):
        image_list = []
        region_images_list = []
        bboxes_list = []
        caps_list = []
        tags_list = []
        ids_list = []
        batch_idx_list = []

        for idx, sample in enumerate(samples):
            image_list.append(sample["image"])
            region_images_list.append(sample["region_images"])
            bboxes_list.append(sample["bboxes"])
            caps_list.extend(sample["caps"])
            tags_list.append(sample["tags"])
            ids_list.extend(sample["ids"])
            batch_idx_list.extend([idx] * len(sample["bboxes"]))

        return {
            "image": torch.stack(image_list, dim=0),
            "region_images": torch.cat(region_images_list, dim=0),
            "bboxes": torch.cat(bboxes_list, dim=0),
            "caps": caps_list,
            "tags": torch.cat(tags_list, dim=0),
            "ids": ids_list,
            "batch_idx": torch.LongTensor(batch_idx_list),
        }


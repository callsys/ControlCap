import os
# import json
import copy
import ujson as json
import concurrent.futures as futures
import sng_parser
import tqdm
import cv2
import sys
import shutil
import torch
from torchvision.ops import box_iou
import logging
import numpy as np
from functools import partial
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from itertools import groupby
from PIL import Image
import multiprocessing
import matplotlib.pyplot as plt

sys.path.append("../")

# from models.func_ram import RAM
# from models.func_gdino import GDINO
# from models.func_sam import SAM, rle_to_mask
# from models.func_sam import rle_to_mask




class BaseProcessor:
    def __init__(self,
                 version="",
                 image_root="",
                 ann_root="",
                 save_file=""):
        self.version = version
        self.image_root = image_root
        self.ann_root = ann_root
        self.save_file = save_file
        self.default_batch_size = 1
        self.batch_size = 1
        with open("controlcap/common/data/assets/ram_tag_list.txt", "r") as fr:
            tag_list = fr.readlines()
        self.tag_list = [tag.strip().lower() for tag in tag_list]
        self.id2tag = dict()
        self.tag2id = dict()
        for id, tag in enumerate(self.tag_list):
            self.id2tag[id] = tag
            self.tag2id[tag] = id
        with open("controlcap/common/data/assets/ram_tag_list_threshold.txt", "r") as fr:
            tag_list_thr = fr.readlines()
        self.tag_list_thr = [float(thr) for thr in tag_list_thr]
        self.tag_list_thr_high = [float(thr)*1.4 for thr in tag_list_thr]

        # multiprocessing.set_start_method('spawn')

        self.gpu_ids = list(range(torch.cuda.device_count()))

    def visualize(self, file=None, save_dir=None):
        if file is None:
            file = self.save_file
        if save_dir is None:
            save_dir = "./viz"
        print(f"[INFO]:\tvisualize ({file})")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        os.mkdir(save_dir)
        coco = COCO(file)
        image_root = coco.dataset["dataset"]["image_root"]
        imgs = coco.imgs
        imgs = list(imgs.items())
        expand_ratio = 10
        max_num = 50
        count = 0
        for img_id, img in tqdm.tqdm(imgs[:max_num]):
            img_path = os.path.join(image_root, img["file_name"])
            anns = coco.imgToAnns.get(img_id, [])
            if len(anns) == 0:
                continue
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            for ann in anns[:1]:
                cimage = copy.deepcopy(image)
                draw_caption = []
                caption = ann["caption"]
                draw_caption.append(f"Caption: [{caption.lower()}]")

                extra_info = ann["extra_info"]
                if "parser_result" in extra_info:
                    tag_set0 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set0"]]
                    tag_set1 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set1"]]
                    tag_set2 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set2"]]
                    draw_caption.append("Parse Tag: [" + ",".join(tag_set0) + "][" + ",".join(tag_set1) + "][" + ",".join(tag_set2) + "]")
                if "ram_mask_result" in extra_info:
                    tag_set = [self.tag_list[i] for i in extra_info["ram_mask_result"]["tag_set"]]
                    draw_caption.append("RAM mask Tag: [" + ",".join(tag_set) + "]")
                if "ram_bbox_result" in extra_info:
                    tag_set = [self.tag_list[i] for i in extra_info["ram_bbox_result"]["tag_set"]]
                    draw_caption.append("RAM bbox Tag: [" + ",".join(tag_set) + "]")
                if "pp_result" in extra_info:
                    tag_set1 = extra_info["pp_result"]["tag_set1"]
                    tag_set2 = extra_info["pp_result"]["tag_set2"]
                    draw_caption.append("PostProcess Tag: [" + ",".join(tag_set1) + "][" + ",".join(tag_set2) + "]")

                if "bbox" in ann:
                    bbox = ann["bbox"]
                    seg = None
                else:
                    continue

                if "sam_result" in extra_info:
                    rle = extra_info["sam_result"]["segmentation"]
                    seg = rle_to_mask(rle)

                bbox = [int(i) for i in bbox]

                pos = [bbox[0] * expand_ratio, (bbox[1]-2) * expand_ratio]

                rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
                rgb = [int(el) for el in rgb[0]]
                cv2.rectangle(cimage, [bbox[0], bbox[1]], [bbox[2], bbox[3]], color=rgb, thickness=1)

                if seg is not None:
                    seg_image = seg.reshape(h, w, 1) * np.array(rgb).reshape(1, 1, -1)
                    cimage[seg] = cimage[seg] * 0.5 + seg_image[seg] * 0.5


                captions_to_draw = [(draw_caption, pos, rgb)]

                dsize = (w * expand_ratio, h * expand_ratio)
                cimage = cv2.resize(cimage, dsize)

                for draw_caption, pos, rgb in captions_to_draw:
                    for cap in draw_caption:
                        cv2.putText(cimage, cap, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=rgb, thickness=3)
                        pos[1] = pos[1] - int(3.5 * expand_ratio)

                save_path = os.path.join(save_dir, os.path.basename(img["file_name"]))
                cv2.imwrite(save_path, cimage)
                count = count + 1
                if count >= max_num:
                    break
            if count >= max_num:
                break

        print(f"[INFO]:\tsave to ({save_dir})")
        return save_dir

    def process_wrapper(self, args):
        total_processes = args.get("max_workers", 1)
        idx = args.get("pid", 0)
        anns = args.get("anns", [])
        imgs = args.get("imgs", [])
        task_func = args.get("task_func", None)
        gpu_id = self.gpu_ids[idx % len(self.gpu_ids)]
        if "model" in args:
            model = args.get("model")
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            gpu_model = model(device=device)
            task_func = partial(task_func, model=gpu_model)

        if len(anns)>0:
            banns = [[ann[0] for ann in bann[1]] for bann in groupby(zip(anns, range(len(anns))), key = lambda x:x[1]//self.batch_size)]
            parsed_anns = []
            for bann in tqdm.tqdm(banns, desc=f"Process {idx + 1}/{total_processes}"):
                try:
                    parsed_bann = task_func(bann=bann)
                    parsed_anns.extend(parsed_bann)
                except:
                    print(f"ERROR sample on cuda:{gpu_id}")
                    raise
                    parsed_anns.extend(bann)
            return parsed_anns
        else:
            assert len(imgs)>0
            bimgs = [[img[0] for img in bimg[1]] for bimg in
                     groupby(zip(imgs, range(len(imgs))), key=lambda x: x[1] // self.batch_size)]
            parsed_anns = []
            parsed_imgs = []
            for bimg in tqdm.tqdm(bimgs, desc=f"Process {idx + 1}/{total_processes}"):
                try:
                    result = task_func(bimg=bimg)
                    if len(result)>0 and "file_name" in result[0]:
                        parsed_imgs.extend(result)
                    else:
                        parsed_anns.extend(result)
                except:
                    print(f"ERROR sample on cuda:{gpu_id}")
                    if len(parsed_imgs)>0:
                        parsed_imgs.extend(bimg)
                    else:
                        img_ids = [img["id"] for img in bimg]
                        anns = []
                        for img_id in img_ids:
                            anns.extend(self.coco.imgToAnns[img_id])
                        parsed_anns.extend(anns)
            if len(parsed_imgs) == 0 and len(parsed_anns) == 0:
                return bimgs
            elif len(parsed_imgs)>0:
                return parsed_imgs
            else:
                return parsed_anns

    def parse_graph_single_process(self, bann):
        parsed_bann = []
        for ann in bann:
            caption = ann["caption"]
            graph = sng_parser.parse(caption)
            result = {"graph": graph}
            ann["extra_info"]["parser_result"] = result
            parsed_bann.append(ann)
        return parsed_bann

    def parse_tag_single_process(self, bann):
        parsed_bann = []
        for ann in bann:
            graph = ann["extra_info"]["parser_result"]["graph"]

            extra_info = ann["extra_info"]
            tag_set0 = []
            if "grit_result" in extra_info:
                noun = extra_info["grit_result"]["noun"].lower()
                graph = sng_parser.parse(noun)
                entities = graph["entities"]
                relations = graph["relations"]
                for relation in relations:
                    tag_set0.append(relation["relation"])
                for entity in entities:
                    tag_set0.append(entity["head"])
                    lemma_span = entity["lemma_span"]
                    tokens = lemma_span.split(" ")
                    for l in range(len(tokens)):
                        tag = " ".join(tokens[-l - 1:])
                        tag_set0.append(tag)
                    for modifier in entity["modifiers"]:
                        if modifier["dep"] == "det":
                            continue
                        tag_set0.append(modifier["span"])

            tag_set1 = []
            tag_set2 = []
            entities = graph["entities"]
            relations = graph["relations"]
            objects = []
            for relation in relations:
                objects.append(relation["object"])
                tag_set2.append(relation["relation"])
            objects = set(objects)
            subjects = set(list(range(len(entities)))) - objects

            for id in subjects:
                entity = entities[id]
                tag_set1.append(entity["head"])
                lemma_span = entity["lemma_span"]
                tokens = lemma_span.split(" ")
                for l in range(len(tokens)):
                    tag = " ".join(tokens[-l - 1:])
                    tag_set1.append(tag)
                for modifier in entity["modifiers"]:
                    if modifier["dep"] == "det":
                        continue
                    tag_set1.append(modifier["span"])

            for id in objects:
                entity = entities[id]
                tag_set2.append(entity["head"])
                lemma_span = entity["lemma_span"]
                tokens = lemma_span.split(" ")
                for l in range(len(tokens)):
                    tag = " ".join(tokens[-l - 1:])
                    tag_set2.append(tag)
                for modifier in entity["modifiers"]:
                    if modifier["dep"] == "det":
                        continue
                    tag_set2.append(modifier["span"])

            tag_set0 = set(tag.lower().strip() for tag in tag_set0 if tag in self.tag_list)
            tag_set1 = set(tag.lower().strip() for tag in tag_set1 if tag in self.tag_list)
            tag_set2 = set(tag.lower().strip() for tag in tag_set2 if tag in self.tag_list)

            tag_set0 = [self.tag2id[i] for i in tag_set0]
            tag_set1 = [self.tag2id[i] for i in tag_set1]
            tag_set2 = [self.tag2id[i] for i in tag_set2]

            ann["extra_info"]["parser_result"]["tag_set0"] = tag_set0
            ann["extra_info"]["parser_result"]["tag_set1"] = tag_set1
            ann["extra_info"]["parser_result"]["tag_set2"] = tag_set2
            parsed_bann.append(ann)
        return parsed_bann

    def post_process_single_process(self, bann):
        parsed_bann = []
        for ann in bann:
            caption = ann["caption"]

            tag_set1 = dict()
            tag_set2 = dict()
            extra_info = ann["extra_info"]
            if "parser_result" in extra_info:
                parser_tag_set0 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set0"]]
                parser_tag_set1 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set1"]]
                parser_tag_set2 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set2"]]
                for tag in parser_tag_set0:
                    tag_set1[tag] = tag_set1.get(tag, "") + "(p0)"
                    tag_set2[tag] = tag_set2.get(tag, "") + "(p0)"
                for tag in parser_tag_set1:
                    tag_set1[tag] = tag_set1.get(tag, "") + "(p1)"
                    tag_set2[tag] = tag_set2.get(tag, "") + "(p1)"
                for tag in parser_tag_set2:
                    tag_set2[tag] = tag_set2.get(tag, "") + "(p2)"

            if "gdino_mask_result" in extra_info:
                tag_set = [self.tag_list[i] for i in extra_info["gdino_mask_result"]["tag_set"]]
                bboxes = torch.Tensor(extra_info["gdino_mask_result"]["bboxes"])
                gt_bbox = torch.Tensor(ann["bbox"])
                ious = box_iou(gt_bbox[None], bboxes)
                idx = torch.where(ious>0.7)[0]
                ground_tag_set = tag_set[idx] if len(idx)>0 else []
                ground_tag_set = [ground_tag_set] if not isinstance(ground_tag_set, list) else ground_tag_set
                for tag in ground_tag_set:
                    tag_set2[tag] = tag_set2.get(tag, "") + "(gb)"

            if "ram_mask_result" in extra_info:
                tag_score = extra_info["ram_mask_result"]["tag_score"]
                tag_set = extra_info["ram_mask_result"]["tag_set"]
                ram_mask_tag_set = [self.tag_list[id] for id, score in zip(tag_set, tag_score) if
                                    score>self.tag_list_thr_high[id]]
                for tag in ram_mask_tag_set:
                    tag_set1[tag] = tag_set1.get(tag, "") + "(rm)"
                    tag_set2[tag] = tag_set2.get(tag, "") + "(rm)"

            if "ram_bbox_result" in extra_info:
                tag_score = extra_info["ram_bbox_result"]["tag_score"]
                tag_set = extra_info["ram_bbox_result"]["tag_set"]
                ram_bbox_tag_set = [self.tag_list[id] for id, score in zip(tag_set, tag_score) if
                                    score > self.tag_list_thr_high[id]]
                for tag in ram_bbox_tag_set:
                    tag_set2[tag] = tag_set2.get(tag, "") + "(rb)"

            tag_set1 = [k + v for k, v in tag_set1.items()]
            tag_set2 = [k + v for k, v in tag_set2.items()]
            result = {"tag_set1": tag_set1, "tag_set2": tag_set2}

            ann["extra_info"]["pp_result"] = result

            parsed_bann.append(ann)
        return parsed_bann

    def parse_graph(self, file=None, max_workers=32, batch_size=None):
        if file is None:
            file = self.save_file
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        print(f"[INFO]:\tparse graph for ({file})")
        json_file = json.load(open(file, "r"))
        anns = json_file["annotations"]

        process_args = [{"anns": anns[i::max_workers],
                         "pid": i,
                         "max_workers": max_workers,
                         "task_func": self.parse_graph_single_process} for i in range(max_workers)]

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_wrapper, process_args))

        parsed_anns = []
        for result in results:
            parsed_anns.extend(result)

        json_file["annotations"] = parsed_anns
        print(f"[INFO]:\tdump annotation to ({file})")
        with open(file, "w") as fw:
            json.dump(json_file, fw)
        return file

    def parse_tags(self, file=None, max_workers=32, batch_size=None):
        if file is None:
            file = self.save_file
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        print(f"[INFO]:\tparse tags for ({file})")
        json_file = json.load(open(file, "r"))
        anns = json_file["annotations"]

        process_args = [{"anns": anns[i::max_workers],
                         "pid": i,
                         "max_workers": max_workers,
                         "task_func": self.parse_tag_single_process} for i in range(max_workers)]

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_wrapper, process_args))

        parsed_anns = []
        for result in results:
            parsed_anns.extend(result)

        json_file["annotations"] = parsed_anns
        print(f"[INFO]:\tdump annotation to ({file})")
        with open(file, "w") as fw:
            json.dump(json_file, fw)
        return file

    def post_process(self, file=None, max_workers=1, batch_size=None):
        if file is None:
            file = self.save_file
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        print(f"[INFO]:\tpost process for ({file})")
        json_file = json.load(open(file, "r"))
        anns = json_file["annotations"]

        process_args = [{"anns": anns[i::max_workers],
                         "pid": i,
                         "max_workers": max_workers,
                         "task_func": self.post_process_single_process} for i in range(max_workers)]

        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_wrapper, process_args))

        parsed_anns = []
        for result in results:
            parsed_anns.extend(result)

        json_file["annotations"] = parsed_anns
        print(f"[INFO]:\tdump annotation to ({file})")
        with open(file, "w") as fw:
            json.dump(json_file, fw)
        return file

    def summary_anns(self, file=None):
        if file is None:
            file = self.save_file
        self.coco = json.load(open(file, "r"))
        # self.coco = COCO(file)

        return
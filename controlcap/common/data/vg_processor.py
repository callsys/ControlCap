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


class VGProcessor(BaseProcessor):
    def __init__(self,
                version = "vg1.2",
                image_root = "data/vg/images",
                ann_root = "data/vg/annotations/vg1.2",
                save_dir = ""):
        super().__init__()
        self.version = version
        self.image_root = image_root
        self.ann_root = ann_root

        if not os.path.exists(os.path.join(save_dir)):
            os.system(f"mkdir -p {save_dir}")

        save_files = []

        if version == "vg_reg":
            splits = ["test"]
        else:
            splits = ["train", "val", "test"]
            # splits = ["test"]

        for split in splits:
            save_files.append(os.path.join(save_dir, f"{split}.json"))
        self.save_files = save_files
        self.save_file = ""

        self.UNK_IDENTIFIER = '<unk>'
        self.MAX_WORDS = 10

        self.region_descriptions = os.path.join(ann_root, "region_descriptions.json")
        self.image_data = os.path.join(ann_root, "image_data.json")
        self.densecap_splits = os.path.join(ann_root, "densecap_splits.json")
        self.attributes = os.path.join(ann_root, "attributes.json")
        self.synsets = os.path.join(ann_root, "synsets.json")
        self.objects = os.path.join(ann_root, "objects.json")
        self.region_graphs = os.path.join(ann_root, "region_graphs.json")
        # self.scene_graphs = os.path.join(ann_root, "scene_graphs.json")
        # self.relationships = os.path.join(ann_root, "relationships.json")
        self.vocabulary_size = 10000  # 10497#from dense caption paper
        self.HAS_VOCAB = True

        self.images_num = 0
        self.captions_num = 0
        self.skip_captions_num = 0
        self.num_invalid_bbox = 0
        self.num_empty_phrase = 0

    def init_vocabulary(self, phrases_all):
        words_to_count = {}
        word_freq = Counter(itertools.chain(*phrases_all))
        print(f"Found {len(word_freq.items())} unique word tokens.")
        vocab_freq = word_freq.most_common(self.vocabulary_size-1)
        self.vocabulary_inverted = [x[0] for x in vocab_freq]
        self.vocabulary_inverted.insert(0,self.UNK_IDENTIFIER)
        print(f"Using vocabulary size {self.vocabulary_size}.")
        print(f"The least frequent word in our vocabulary is '{vocab_freq[-1][0]}' and appeared {vocab_freq[-1][1]} times.")

    def dump_vocabulary(self, vocab_filename):
        print(f'Dumping vocabulary to file: {vocab_filename}')
        with open(vocab_filename, 'wb') as vocab_file:
            for word in self.vocabulary_inverted:
                vocab_file.write(f'{word}\n')
        print('Done.')

    def word_preprocess(self, phrase):
        """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
        replacements = {
            u'½': u'half',
            u'—': u'-',
            u'™': u'',
            u'¢': u'cent',
            u'ç': u'c',
            u'û': u'u',
            u'é': u'e',
            u'°': u' degree',
            u'è': u'e',
            u'…': u'',
        }
        for k, v in replacements.items():
            phrase = phrase.replace(k, v)
        translation_table = str.maketrans("", "", string.punctuation)
        tokens = str(phrase).lower().translate(translation_table).split()
        return tokens

    def box_preprocess(self, obj, image_height, image_width):
        x1, y1, x2, y2 = obj["x"], obj["y"], obj["x"]+obj["width"], obj["y"]+obj["height"]
        # clamp to image
        valid = True
        if x1 < 1:
            x1 = 1
        if y1 < 1:
            y1 = 1
        if x1 > image_width - 1:
            x1 = image_width - 1
        if y1 > image_height - 1:
            y1 = image_height - 1
        if x2 > image_width:
            x2 = image_width
        if y2 > image_height:
            y2 = image_height

        try:
            assert x2 - x1 > 0
            assert y2 - y1 > 0
        except:
            valid = False

        bbox = [x1, y1, x2, y2]
        return valid, bbox

    def filter_images(self, regions_all, image_data, split_image_ids):
        filter_regions_all = []
        filter_image_data = []
        for img, idata in zip(regions_all, image_data):
            keep = img["id"] in split_image_ids and len(img['regions']) > 0
            if idata["coco_id"] is not None and idata["coco_id"] in self.exclude_ids:
                keep = False
            if self.version == "vgcoco" and idata["coco_id"] is None:
                keep = False
            if keep:
                filter_regions_all.append(img)
                filter_image_data.append(idata)
        return filter_regions_all, filter_image_data

    def parse_anns(self):
        if "train" in self.save_file:
            split_name = "train"
        elif "val" in self.save_file:
            split_name = "val"
        else:
            split_name = "test"
        dataset_sample = {"dataset":{"description": self.version,
                                  "image_root": self.image_root,
                                  "extra_info": dict()},
                          "images":[],
                          "annotations":[]}

        print("=" * 50)
        split = json.load(open(self.densecap_splits, "r"))
        split_image_ids = split[split_name]
        print(f'split image number: {len(split_image_ids)}')
        regions_all = json.load(open(self.region_descriptions))
        image_data = json.load(open(self.image_data))

        regions_all, image_data = self.filter_images(regions_all, image_data, split_image_ids)

        num_bbox = 0
        num_empty_phrase = 0
        num_invalid_bbox = 0
        ann_id = 1
        for item, image_info in zip(tqdm.tqdm(regions_all), image_data):
            im_id = item['id']
            if self.version == "vg1.0":
                image_id = image_info["id"]
            else:
                image_id = image_info["image_id"]
            if im_id != image_id:
                print('region and image metadata inconsistent')
                exit()
            # tokenize phrase
            num_bbox += len(item['regions'])
            regions_filt = []

            image = f"{im_id}.jpg"
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            image_h, image_w, _ = cv2.imread(abs_image_path).shape
            image_sample = {"id": im_id,
                            "file_name": image,
                            "height": image_h,
                            "width": image_w,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            for region in item['regions']:
                extra_info = {"vg_result": dict()}
                phrase = region['phrase']
                tokens = self.word_preprocess(phrase)
                if not len(tokens) <= 15:
                    self.skip_captions_num += 1
                    continue
                phrase = " ".join(tokens)

                if len(phrase) == 0:
                    if split_name == "train":
                        continue
                    num_empty_phrase += 1

                regions_filt.append(region)

                valid, bbox = self.box_preprocess(region, image_h, image_w)

                if not valid:
                    if split_name == "train":
                        continue
                    num_invalid_bbox += 1

                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann_sample = {"id": ann_id,
                              "image_id": im_id,
                              "caption": phrase,
                              "bbox": bbox,
                              "segmentation": seg,
                              "extra_info": extra_info}
                dataset_sample["annotations"].append(ann_sample)
                ann_id = ann_id + 1
        non_empty_image_ids = set([ann["image_id"] for ann in dataset_sample["annotations"]])
        images = dataset_sample["images"]
        filter_images = [image for image in images if image["id"] in non_empty_image_ids]
        dataset_sample["images"] = filter_images

        self.images_num = self.images_num + len(dataset_sample["images"])
        self.captions_num = self.captions_num + len(dataset_sample["annotations"])
        self.num_empty_phrase = self.num_empty_phrase + num_empty_phrase
        self.num_invalid_bbox = self.num_invalid_bbox + num_invalid_bbox

        print(f"[INFO]:\t{len(dataset_sample['annotations'])} annotations")
        print(f"[INFO]:\t{len(dataset_sample['images'])} images")
        print(f"[INFO]:\t{num_empty_phrase} empty phrase")
        print(f"[INFO]:\t{num_invalid_bbox} invalid bbox")
        print(f"[INFO]:\t({self.version}) dump annotation to ({self.save_file})")
        with open(self.save_file, "w") as fw:
            json.dump(dataset_sample, fw)

        return self.save_file

    def parse_anns_reg(self):
        dataset_sample = {"dataset":{"description": self.version,
                                  "image_root": self.image_root,
                                  "extra_info": dict()},
                          "images":[],
                          "annotations":[]}

        gt_path = self.ann_root
        gt = COCO(gt_path)
        imgs = gt.imgs

        for img_id, img in tqdm.tqdm(imgs.items()):
            image = img["file_name"]
            abs_image_path = os.path.join(self.image_root, image)
            assert os.path.exists(abs_image_path)
            # height, width = img["height"], img["width"]
            image_h, image_w, _ = cv2.imread(abs_image_path).shape
            image_sample = {"id": img_id,
                            "file_name": image,
                            "height": image_h,
                            "width": image_w,
                            "extra_info": dict()}
            dataset_sample["images"].append(image_sample)

            anns = gt.imgToAnns[img_id]
            for ann in anns:
                bbox = ann["bbox"]
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]

                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann_sample = {"id": ann["id"],
                              "image_id": img_id,
                              "caption": ann["caption"],
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

    def post_process_single_process(self, bann):
        parsed_bann = []
        for ann in bann:
            caption = ann["caption"]

            tag_set1 = dict()
            tag_set2 = dict()

            extra_info = ann["extra_info"]

            if "vg_result" in extra_info:
                objects = extra_info["vg_result"]["objects"]
                for object in objects:
                    names = object.get("names", [])
                    attributes = object.get("attributes", [])
                    for tag in names:
                        if tag in self.tag_list:
                            tag_set1[tag] = tag_set1.get(tag, "") + "(p0)"
                            tag_set2[tag] = tag_set2.get(tag, "") + "(p0)"
                    for tag in attributes:
                        if tag in self.tag_list:
                            tag_set1[tag] = tag_set1.get(tag, "") + "(p0)"
                            tag_set2[tag] = tag_set2.get(tag, "") + "(p0)"

            if "parser_result" in extra_info:
                # parser_tag_set0 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set0"]]
                parser_tag_set1 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set1"]]
                parser_tag_set2 = [self.tag_list[i] for i in extra_info["parser_result"]["tag_set2"]]
                # for tag in parser_tag_set0:
                #     tag_set1[tag] = tag_set1.get(tag, "") + "(p0)"
                #     tag_set2[tag] = tag_set2.get(tag, "") + "(p0)"
                for tag in parser_tag_set1:
                    tag_set1[tag] = tag_set1.get(tag, "") + "(p1)"
                    tag_set2[tag] = tag_set2.get(tag, "") + "(p1)"
                for tag in parser_tag_set2:
                    tag_set2[tag] = tag_set2.get(tag, "") + "(p2)"

            if "ram_bbox_result" in extra_info:
                tag_score = extra_info["ram_bbox_result"]["tag_score"]
                tag_set = extra_info["ram_bbox_result"]["tag_set"]
                ram_bbox_tag_set = [self.tag_list[id] for id, score in zip(tag_set, tag_score) if
                                    score > self.tag_list_thr_high[id]]
                for tag in ram_bbox_tag_set:
                    if tag in caption:
                        tag_set1[tag] = tag_set1.get(tag, "") + "(rb)"
                    tag_set2[tag] = tag_set2.get(tag, "") + "(rb)"

            tag_set1 = [k + v for k, v in tag_set1.items()]
            tag_set2 = [k + v for k, v in tag_set2.items()]
            result = {"tag_set1": tag_set1, "tag_set2": tag_set2}

            ann["extra_info"]["pp_result"] = result

            parsed_bann.append(ann)
        return parsed_bann

    def entropy(self, image1):
        hist_1d, _ = np.histogram(image1.ravel(), bins=128)
        px = hist_1d / float(np.sum(hist_1d))
        nzs = px > 0
        return -np.sum(px[nzs] * np.log(px[nzs]))

    def mutual_info(self, image1, image2):
        hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=128)
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

    def cal_region_entropy(self, image, bbox, target_bbox):
        image_arr = np.array(image)
        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
        res = 112

        x1, y1, x2, y2 = np.array(bbox).astype(np.int64)
        region_arr = cv2.resize(image_arr[y1:y2, x1:x2], (res, res))

        x1, y1, x2, y2 = np.array(target_bbox).astype(np.int64)
        target_arr = cv2.resize(image_arr[y1:y2, x1:x2], (res, res))

        image_arr = cv2.resize(image_arr, (res, res))

        info = self.entropy(region_arr) - \
                   self.mutual_info(region_arr, target_arr) - \
                   self.mutual_info(region_arr, image_arr)

        # infos = []
        # for i in range(3):
        #     info = self.entropy(region_arr[..., i]) - \
        #            self.mutual_info(region_arr[..., i], target_arr[..., i]) - \
        #            self.mutual_info(region_arr[..., i], image_arr[..., i])
        #
        #     infos.append(info)

        return info

    def AinB(self, bbox1, bbox2):
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2

        if xmin1>xmin2 and ymin1>ymin2 and xmax1<xmax2 and ymax1<ymax2:
            return 1
        return 0

    def run_info_bboxes_single_process(self, bann=None):
        results = []
        for ann in bann:
            image_id = ann["image_id"]
            img = self.coco.imgs[image_id]
            image = os.path.join(self.coco.dataset["dataset"]["image_root"], img["file_name"])
            image = cv2.imread(image)

            info_bboxes = []
            h, w, c = image.shape
            info_bboxes.append([0, 0, w, h])

            bbox = ann["bbox"]
            min_size = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
            info_bboxes.append(bbox)

            candi_bboxes = []

            # scale = 224
            # img_lbl, regions = selectivesearch.selective_search(
            #     copy.deepcopy(image), scale=scale, sigma=0.9, min_size=min_size)
            #
            # # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            #
            # for r in regions:
            #     sx, sy, sh, sw = r['rect']
            #     candi_bbox = [sx, sy, sx+sw, sy+sh]
            #     candi_bbox = [int(i) for i in candi_bbox]
            #
            #     # candi_bboxes.append(candi_bbox)
            #     # cv2.rectangle(image, (candi_bbox[0], candi_bbox[1]), (candi_bbox[2], candi_bbox[3]), (0, 0, 255), 2)
            #
            #     if self.AinB(bbox, candi_bbox):
            #         candi_bboxes.append(candi_bbox)
            #         cv2.rectangle(image, (candi_bbox[0], candi_bbox[1]), (candi_bbox[2], candi_bbox[3]), (0, 255, 0), 2)

            num_views = 10
            weights = [((i + 1) / num_views+1) ** 2 for i in range(num_views)]
            oowh = np.array([0, 0, w, h]).astype(np.float32)
            for we in weights:
                bbox_arr = np.array(bbox).astype(np.float32)
                candi_bbox = bbox_arr + (oowh - bbox_arr) * we
                candi_bboxes.append([int(i) for i in candi_bbox])

            infos = []
            for candi_bbox in candi_bboxes:
                info = self.cal_region_entropy(image, candi_bbox, bbox)
                infos.append(info)
            idx = np.argmax(infos)
            info_bbox = candi_bboxes[idx]
            info_bboxes.append(info_bbox)

            results.append({"bboxes": info_bboxes})

            # cv2.rectangle(image, (info_bbox[0], info_bbox[1]), (info_bbox[2], info_bbox[3]), (0, 0, 255), 2)
            # plt.imshow(image)

            # print(0)

        for ann, result in zip(bann, results):
            ann["extra_info"]["info_result"] = result
        return bann

    def run_info_bboxes(self, file=None, max_workers=8, batch_size=None):
        if file is None:
            file = self.save_file
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        print(f"[INFO]:\trun sam for ({file})")
        json_file = json.load(open(file, "r"))
        self.coco = COCO(file)

        anns = json_file["annotations"]

        process_args = [{"anns": anns[i::max_workers],
                         "pid": i,
                         "max_workers": max_workers,
                         "task_func": self.run_info_bboxes_single_process} for i in range(max_workers)]

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

    def process(self):
        for save_file in self.save_files:
            self.save_file = save_file
            if "reg" in save_file:
                save_file = self.parse_anns_reg()
            else:
                save_file = self.parse_anns()
            save_file = self.parse_graph(save_file)
            save_file = self.parse_tags(save_file)
            self.run_info_bboxes(save_file)
            # save_file = self.post_process(save_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual Genome (VG) data processor")
    parser.add_argument("--version", choices=["vg1.2", "vg1.0", "vgcoco", "vg_reg"], default="vg1.2", help="dataset version.")
    parser.add_argument("--image-root", default="data/vg/images", help="path to image root.")
    parser.add_argument("--ann-root", default="data/vg/annotations/vg1.2", help="path to annotation root.")
    parser.add_argument("--save-dir", default="data/vg/controlcap/", help="path to save dir.")
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
    processor = VGProcessor(version=args.version,
                            image_root=args.image_root,
                            ann_root=args.ann_root,
                            save_dir=args.save_dir)
    processor.process()
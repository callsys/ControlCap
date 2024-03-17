import torch
import numpy as np
import cv2
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from omegaconf import OmegaConf
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import InterpolationMode

from lavis.common.registry import registry
from lavis.processors.blip_processors import BlipImageBaseProcessor

@registry.register_processor("controlcap")
class ControlCapProcessor(BlipImageBaseProcessor):
    def __init__(
            self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        region_transform = [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToImage(),
                # transforms.ToImageTensor(),
                transforms.ConvertImageDtype(),
                transforms.Normalize(mean, std),
            ]

        self.transform = transforms.Compose(
            region_transform
        )

        seg_transform = [
            transforms.Resize(
                (image_size, image_size), interpolation=InterpolationMode.BICUBIC
            ),
        ]

        self.seg_transform = transforms.Compose(
            seg_transform
        )

        # if False, transform the arbitrary mask into bbox
        self.with_seg = True

    def seq2mask(self, image, segs):
        w, h = image.size
        bboxes = []
        masks = []
        for seg in segs:
            if isinstance(seg, list):
                seq = []
                for seg_ in seg:
                    seq.extend(seg_)
                x1, y1 = np.array(seq).reshape(-1, 2).min(0)
                x2, y2 = np.array(seq).reshape(-1, 2).max(0)
                if x1 >= x2 and x2 - 0 > 0:
                    x1 = x2 - 1
                elif x1 >= x2 and w - x1 > 0:
                    x2 = x1 + 1
                if y1 >= y2 and y2 - 0 > 0:
                    y1 = y2 -1
                elif y1 >= y2 and h - y1 > 0:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                mask = np.zeros((h, w), np.uint8)
                for seg_ in seg:
                    mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                bboxes.append(bbox)
                masks.append(mask)
            else:
                if isinstance(seg["counts"], list):
                    seg = mask_util.frPyObjects(seg, *seg["size"])
                elif not isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].encode()
                mask = mask_util.decode(seg)
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
                bboxes.append(bbox)
                masks.append(mask)

        if not self.with_seg:
            masks = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                seg = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                mask = np.zeros((h, w), np.uint8)
                for seg_ in seg:
                    mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                masks.append(mask)
        return bboxes, masks

    def region_process(self, image, bboxes, segs):
        w, h = image.size
        region_images = []
        region_segs = []
        for bbox, seg in zip(bboxes, segs):
            x1, y1, x2, y2 = [int(el) for el in bbox]
            region = [max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)]
            region_image = Image.fromarray(np.array(image)[region[1]:region[3], region[0]:region[2]])
            region_seg = Image.fromarray(np.array(seg)[region[1]:region[3], region[0]:region[2]])
            region_images.append(region_image)
            region_segs.append(region_seg)
        region_images = self.transform(region_images)
        region_images = torch.stack(region_images, 0)
        region_segs = self.seg_transform(region_segs)
        region_segs = torch.from_numpy(np.array([np.array(region_seg) for region_seg in region_segs]))
        return region_images, region_segs

    def image_process(self, image, bboxes, segs):
        w, h = image.size
        bboxes = torch.Tensor(bboxes)
        target = {"boxes": torchvision.tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=(h, w))}
        image, target = self.transform(image, target)
        segs = [Image.fromarray(seg) for seg in segs]
        segs = self.seg_transform(segs)
        segs = torch.from_numpy(np.array([np.array(seg) for seg in segs]))
        bboxes = target["boxes"].data.to(torch.float32)
        return image, bboxes, segs

    def __call__(self, image, segs):
        bboxes, segs = self.seq2mask(image, segs)
        region_images, region_segs = self.region_process(image, bboxes, segs)
        image, bboxes, segs = self.image_process(image, bboxes, segs)

        output = {"image": image,
                  "region_images": region_images,
                  "segs": segs,
                  "region_segs": region_segs,
                  "bboxes": bboxes}

        return output

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


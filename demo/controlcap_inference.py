import torch
from transformers import AutoTokenizer, CLIPImageProcessor
from functools import partial
import os
import numpy as np
import cv2
from PIL import Image

import lavis.tasks as tasks
from lavis.common.registry import registry
from controlcap.common.config import Config


def show_mask(mask, image, random_color=True, img_trans=0.9, mask_trans=0.5, return_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3) * 255], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    image = cv2.addWeighted(image, img_trans, mask_image.astype('uint8'), mask_trans, 0)
    if return_color:
        return image, mask_image
    else:
        return image

class ControlCap():
    def __init__(self, args, device='cuda'):
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg)

        builder = registry.get_builder_class("controlcap")(cfg.datasets_cfg.vg_reg)
        builder.build_processors()

        self.device = device
        self.model = model
        self.vis_processor = builder.vis_processors['eval']
        self.text_processor = builder.text_processors['eval']
        self.model.eval()
        self.model.to(device)

    def seg2seq(self, seg):
        # print(seg.shape)
        x1, x2 = np.nonzero(seg.sum(0) != 0)[0][0], np.nonzero(seg.sum(0) != 0)[0][-1]
        y1, y2 = np.nonzero(seg.sum(1) != 0)[0][0], np.nonzero(seg.sum(1) != 0)[0][-1]
        bbox = [x1, y1, x2, y2]
        print(bbox)
        seq = [x1, y1, x2, y1, x2, y2, x1, y2]
        return [seq]

    def process_controls(self, controls):
        controls = controls.split(",|")
        controls = [control.strip() for control in controls]
        return controls

    def predict(self, image, seg, controls):
        print(controls)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(seg, list):
            seg = self.seg2seq(seg)
        samples = self.vis_processor(image, [seg])
        samples["image"] = samples["image"][None].to(self.device)
        samples["region_images"] = samples["region_images"].to(self.device)
        samples["bboxes"] = samples["bboxes"].to(self.device)
        samples["ids"] = torch.zeros(1).to(torch.int64).to(self.device)
        samples["batch_idx"] = torch.zeros(1).to(torch.int64).to(self.device)
        samples["controls"] = [self.process_controls(controls)]
        with torch.inference_mode():
            output = self.model.predict_answers(samples)
        tag_set1_str = ",".join(output[0]["tag_set1"])
        tag_set2_str = ",".join(output[0]["tag_set2"])
        caption = output[0]["caption"]
        vis_tag = f"({tag_set1_str}) ({tag_set2_str})"
        vis_cap = f"{caption}"
        return vis_tag, vis_cap

if __name__ == "__main__":
    image_path = "demo/demo.jpg"
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    segs = np.zeros((h, w))
    segs[100:200, 200:400] =1
    # segs = [[0, 0, w, 0, w, h, 0, h]]
    controlcap = ControlCap()
    output = controlcap.predict(image, segs)
    print(output)
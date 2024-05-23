import copy
import sys
import torch
import torchvision
from PIL import Image
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import (
    build_sam,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_to_rle_pytorch(tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def rle_to_mask(rle):
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

class SAM:
    def __init__(self,
                 device="cuda",
                 ckpt="assets/sam_vit_h_4b8939.pth"):
        original_stdout = sys.stdout
        sys.stdout = open("nul", "w")
        self.device = device
        self.ckpt = ckpt
        self.sam = build_sam(checkpoint=ckpt).to(device)
        self.resize_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        sys.stdout = original_stdout

    def sam_memory_efficient(self, batched_input):
        total_batch_size = 0
        for input in batched_input:
            total_batch_size = total_batch_size + len(input["boxes"])
        if total_batch_size > 100:
            batched_output = []
            chunked_batched_input = []
            for input in batched_input:
                boxes = input["boxes"]
                if len(boxes) > 100:
                    chunked_boxes = torch.split(boxes, split_size_or_sections=100)
                    for chunked_boxes_ in chunked_boxes:
                        chunked_input = copy.deepcopy(input)
                        chunked_input["boxes"] = chunked_boxes_
                        chunked_batched_input.append(chunked_input)
                else:
                    chunked_batched_input.append(input)
            for input in chunked_batched_input:
                output = self.sam([input], multimask_output=False)
                batched_output.extend(output)
        else:
            batched_output = self.sam(batched_input, multimask_output=False)
        return batched_output

    def process(self, images, bboxes):
        batched_input = []
        for image, bbox in zip(images, bboxes):
            batched_input.append({
                "image": prepare_image(image, self.resize_transform, self.device),
                "boxes": self.resize_transform.apply_boxes_torch(bbox, image.shape[:2]).to(self.device),
                "original_size": image.shape[:2]
            })
        batched_output = self.sam_memory_efficient(batched_input)
        # batched_output = self.sam(batched_input, multimask_output=False)
        results = []
        for output in batched_output:
            for mask in output["masks"]:
                rle = mask_to_rle_pytorch(mask)[0]
                results.append({"segmentation": rle})
        return results


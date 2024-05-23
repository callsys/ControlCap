import sys
import torch
import torchvision
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model.eval()
    model = model.to(device)
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    # model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases

class GDINO:
    def __init__(self,
                 device="cuda",
                 config="assets/GroundingDINO_SwinT_OGC.py",
                 ckpt="assets/groundingdino_swint_ogc.pth"):
        original_stdout = sys.stdout
        sys.stdout = open("nul", "w")
        self.device = device
        self.ckpt = ckpt
        self.box_threshold = 0.25
        self.text_threshold = 0.2
        self.iou_threshold = 0.5
        self.transform = T.Compose([T.RandomResize([800], max_size=1333),
                                    T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.model = load_model(config, ckpt, device=device)
        sys.stdout = original_stdout

    def process(self, images, tag_captions):
        results = []
        for image, caption in zip(images, tag_captions):
            transform_image = self.transform(image, None)[0].to(self.device)
            tag_set = caption.split(", ")
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            with torch.no_grad():
                outputs = self.model(transform_image[None], captions=[caption])
            logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

            # filter output
            logits_filt = logits.clone()
            boxes_filt = boxes.clone()
            filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

            # get phrase
            tokenlizer = self.model.tokenizer
            tokenized = tokenlizer(caption)
            # build pred
            bboxes = []
            ground_tag_set = []
            scores = []
            for logit, bbox in zip(logits_filt, boxes_filt):
                ground_tag = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
                if ground_tag not in tag_set:
                    continue
                bboxes.append(bbox)
                ground_tag_set.append(ground_tag)
                scores.append(logit.max().item())
            if len(bboxes)>0:
                bboxes = torch.stack(bboxes, 0)
                scores = torch.Tensor(scores)

                size = image.size
                H, W = size[1], size[0]
                for i in range(bboxes.size(0)):
                    bboxes[i] = bboxes[i] * torch.Tensor([W, H, W, H])
                    bboxes[i][:2] -= bboxes[i][2:] / 2
                    bboxes[i][2:] += bboxes[i][:2]

                bboxes = bboxes.cpu()
                # use NMS to handle overlapped boxes
                nms_idx = torchvision.ops.nms(bboxes, scores, self.iou_threshold).numpy().tolist()
                bboxes = bboxes[nms_idx].tolist()
                scores = scores[nms_idx].tolist()
                ground_tag_set = [ground_tag_set[idx] for idx in nms_idx]
            result = {"tag_set": ground_tag_set, "bboxes": bboxes, "scores": scores}
            results.append(result)
        return results


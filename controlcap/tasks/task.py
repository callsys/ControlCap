import copy
import logging
import json
import tqdm
import shutil
import os
import sys
import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
from controlcap.commom.evaluation.eval_densecap import DenseCapEvaluator

import lavis.common.dist_utils as dist_utils
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask



@registry.register_task("controlcap")
class ControlCapTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.evaluate = kwargs.get("evaluate", False)
        self.eval_dataset_name = kwargs.get("eval_dataset_name", None)
        self.report_metric = kwargs.get("report_metric", True)
        self.visualize = kwargs.get("visualize", True)

    @classmethod
    def setup_task(cls, cfg):
        return cls(**dict(cfg.run_cfg))

    def train_step(self, model, samples):
        return model(samples)

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_llm", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("loss_tag", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)
                loss_all = loss["loss"]
                loss_llm = loss["loss_llm"] if "loss_llm" in loss else 0.
                loss_tag = loss["loss_tag"] if "loss_tag" in loss else 0.


            # after_train_step()
            if use_amp:
                scaler.scale(loss_all).backward()
            else:
                loss_all.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(loss=loss_all)
            metric_logger.update(loss_llm=loss_llm)
            metric_logger.update(loss_tag=loss_tag)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    def valid_step(self, model, samples):
        return model.predict_answers(samples=samples)

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config)

        load_ckpt_path = cfg.run_cfg.load_ckpt_path
        if load_ckpt_path is not None:
            model.load_checkpoint(url_or_filename=load_ckpt_path)
        return model

    def build_datasets(self, cfg):
        datasets = dict()
        datasets_config = cfg.datasets_cfg
        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        if self.eval_dataset_name is None:
            eval_dataset_name = list(datasets_config)[0]
        else:
            eval_dataset_name = self.eval_dataset_name
            if eval_dataset_name not in datasets_config:
                raise ValueError("Eval dataset name not found.")
        self.eval_dataset_ann_path = datasets_config[eval_dataset_name].build_info.annotations.val[0]

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class("controlcap")(dataset_config)
            dataset = builder.build_datasets()

            if not name == eval_dataset_name:
                dataset.pop("val", None)
                dataset.pop("test", None)

            datasets[name] = dataset

        return datasets

    def save_result(self, result, result_dir, filename, remove_duplicate=""):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("Merging results.")
            result = []
            for rank in tqdm.tqdm(range(get_world_size())):
                result_file_r = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                tmp = json.load(open(result_file_r, "r"))
                result.extend(tmp)

            id2pred = dict()
            for pred in result:
                id = pred.pop("id")
                id2pred[id] = pred

            num_result = 0
            gt = json.load(open(self.eval_dataset_ann_path, "r"))
            annotations = gt["annotations"]
            predictions = []

            for annotation in annotations:
                id = annotation["id"]
                image_id = annotation["image_id"]
                segmentation = annotation["segmentation"]
                extra_info = annotation["extra_info"]
                if id in id2pred:
                    pred = id2pred[id]

                    # add pesudo annotation
                    for cap_type, caps in pred["caption"].items():
                        num_result+=1
                        annotation["caption"][cap_type] = caps

                    # add prediction
                    pred_extra_info = pred.get("extra_info", dict())
                    extra_info.update(pred_extra_info)
                    pred["id"] = id
                    pred["image_id"] = image_id
                    pred["segmentation"] = segmentation
                    pred["extra_info"] = extra_info
                    predictions.append(pred)
            gt["predictions"] = predictions

            result_file = os.path.join(result_dir, filename + ".json")

            with open(result_file, "w") as fw:
                json.dump(gt, fw)

            logging.info(f":Get ({num_result}/{len(annotations)}) predictions.")
            logging.info(f":Save result to ({result_file}).")

        return result_file

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_dir = registry.get_path("result_dir")
        result_file = self.save_result(
            val_result,
            result_dir=result_dir,
            filename=f"{split_name}",
            remove_duplicate="question_id",
        )

        metrics = {"agg_metrics": 0}
        if self.report_metric:
            if ("reg" in self.eval_dataset_name) or ("refcoco" in self.eval_dataset_name):
                self.report_metrics_reg(result_file)
            else:
                # Only support vg evaluation now
                supported_eval_dataset_names = ["vg", "grit"]
                flag = False
                for dataset_name in supported_eval_dataset_names:
                    if dataset_name in self.eval_dataset_name:
                        flag = True
                        break
                if flag:
                    metrics = self.report_metrics_densecap(result_file=result_file)
                else:
                    logging.info(f":Not support evaluation for dataset ({self.eval_dataset_name}).")

        if self.visualize:
            self.visualize_result(result_file, result_dir)

        return metrics

    @dist_utils.main_process
    def visualize_result(self, result_file, result_dir="./", cap_type=None):
        logging.info(f":Begin visualization ({result_file}).")
        save_dir = os.path.join(result_dir, "viz")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        os.mkdir(save_dir)
        file = COCO(result_file)
        dataset_root = file.dataset["info"]["dataset_root"]
        image_root = file.dataset["info"]["image_root"]
        imgs = file.imgs
        imgs = list(imgs.items())
        expand_ratio = 5

        cap_type = "dense"
        if cap_type is None:
            assert "predictions" in file.dataset
            imgToAnns = dict()
            cap_type = list(file.dataset['predictions'][0]['caption'].keys())[0]
            for pred in file.dataset['predictions']:
                image_id = pred['image_id']
                if image_id in imgToAnns:
                    imgToAnns[image_id].append(pred)
                else:
                    imgToAnns[image_id] = [pred]
            file.imgToAnns = imgToAnns

        max_num = 1000
        vis_num = 0

        for image_id, img in tqdm.tqdm(imgs):
            img_path = os.path.join(dataset_root, image_root, img["file_name"])
            anns = file.imgToAnns.get(image_id, [])
            if len(anns) == 0:
                continue
            else:
                vis_num += 1
                if vis_num > max_num:
                    break
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            captions_to_draw = []
            for ann in anns:
                extra_info = ann.get('extra_info', dict())
                vis_caption = ann["caption"][cap_type]
                # vis_caption = file.anns[ann["id"]]['caption']['text']
                if isinstance(vis_caption[0], dict):
                    vis_caption = [el["caption"] for el in vis_caption]
                vis_caption = vis_caption[0]
                if "pred_subj_tags" in extra_info:
                    tag_str = f"stags: {','.join(extra_info['pred_subj_tags'])} | " + f"otags: {','.join(extra_info['pred_obj_tags'])} | "
                    vis_caption = tag_str + vis_caption
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    mask = np.zeros((h, w), np.uint8)
                    for seg_ in seg:
                        mask = cv2.fillPoly(mask, np.array(seg_).reshape(1, -1, 2).astype(np.int64), 1)
                else:
                    if isinstance(seg["counts"], list):
                        seg = mask_util.frPyObjects(seg, *seg["size"])
                    elif not isinstance(seg["counts"], bytes):
                        seg["counts"] = seg["counts"].encode()
                    mask = mask_util.decode(seg)
                # pos = (np.array(np.nonzero(mask)).min(1).astype(np.int64)[1]*expand_ratio,
                #        np.array(np.nonzero(mask)).min(1).astype(np.int64)[0]*expand_ratio)

                x, y, wb, hb = cv2.boundingRect(mask)
                pos = (x*expand_ratio, (y + int(hb/2))*expand_ratio)
                bbox = (x, y, x+wb, y+hb)
                rgb = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
                rgb = [int(el) for el in rgb[0]]
                cv2.rectangle(image, [bbox[0], bbox[1]], [bbox[2], bbox[3]], color=rgb, thickness=1)

                captions_to_draw.append((vis_caption, pos, rgb))

            dsize = (w*expand_ratio, h*expand_ratio)
            image = cv2.resize(image, dsize)

            for caption, pos, rgb in captions_to_draw:
                cv2.putText(image, caption, pos, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=rgb, thickness=3)

            save_path = os.path.join(save_dir, os.path.basename(img["file_name"]))
            cv2.imwrite(save_path, image)

        logging.info(f":Save to ({save_dir}).")

        return

    @dist_utils.main_process
    def report_metrics_densecap(self, result_file):
        logging.info(f":Begin evaluation ({result_file}).")

        def seg2bbox(seg):
            if isinstance(seg, list):
                seq = []
                for seg_ in seg:
                    seq.extend(seg_)
                x1, y1 = np.array(seq).reshape(-1, 2).min(0)
                x2, y2 = np.array(seq).reshape(-1, 2).max(0)
                bbox = [x1, y1, x2, y2]
            else:
                if isinstance(seg["counts"], list):
                    seg = mask_util.frPyObjects(seg, *seg["size"])
                elif not isinstance(seg["counts"], bytes):
                    seg["counts"] = seg["counts"].encode()
                mask = mask_util.decode(seg)
                x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
                y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
                bbox = [x1, y1, x2, y2]
            return bbox

        # prediction
        result = COCO(result_file)
        assert "predictions" in result.dataset
        imgToPreds = dict()
        cap_type = list(result.dataset['predictions'][0]['caption'].keys())[0]
        for pred in result.dataset['predictions']:
            image_id = pred['image_id']
            if image_id in imgToPreds:
                imgToPreds[image_id].append(pred)
            else:
                imgToPreds[image_id] = [pred]
        result.imgToPreds = imgToPreds

        # ground truth
        dataset_root = result.dataset['info']['dataset_root']
        annotation_root = result.dataset['info']['annotation_root']
        gt_file = os.path.join(dataset_root, annotation_root)
        gt = COCO(gt_file)

        empty_pred_num = 0

        # evaluation
        ev = DenseCapEvaluator()
        logging.info(f":Evaluate caption type ({cap_type}).")
        recs = []
        for image_id, _ in tqdm.tqdm(list(gt.imgs.items())):
            # h, w = _["height"], _["width"]
            # scale = 720/max(h, w)
            anns = gt.imgToAnns[image_id]
            rec = dict()
            target_boxes = []
            target_text = []
            for ann in anns:
                box = seg2bbox(ann['segmentation'])
                target_boxes.append(box)
                target_text.append(ann['caption']['dense'][0])
            # target_boxes = [[round(el * scale) for el in el2] for el2 in target_boxes]
            rec['target_boxes'] = target_boxes
            rec['target_text'] = target_text

            preds = result.imgToPreds.get(image_id, [])
            if len(preds) == 0:
                empty_pred_num += 1
                continue
            scores = []
            boxes = []
            text = []
            for pred in preds:
                box = seg2bbox(pred['segmentation'])
                for cap in pred['caption'][cap_type]:
                    pred_score = pred["extra_info"].get("obj_score", 1)
                    cap_score = 1 if isinstance(cap, str) else cap['score']
                    if pred_score == 1:
                        score = cap_score
                    else:
                        score = pred_score
                    cap_caption = cap if isinstance(cap, str) else cap['caption']
                    scores.append(score)
                    boxes.append(box)
                    text.append(cap_caption)
            # boxes = [[round(el * scale) for el in el2] for el2 in boxes]
            rec['scores'] = scores
            rec['boxes'] = boxes
            rec['text'] = text

            rec['img_info'] = image_id
            recs.append(rec)

        for rec in tqdm.tqdm(recs):
            try:
                ev.add_result(
                    scores=torch.tensor(rec['scores']),
                    boxes=torch.tensor(rec['boxes']),
                    text=rec['text'],
                    target_boxes=torch.tensor(rec['target_boxes']),
                    target_text=rec['target_text'],
                    img_info=rec['img_info'],
                )
            except:
                print("sample error")

        if empty_pred_num != 0:
            logging.info(f":Image numbers with empty prediction ({empty_pred_num}).")

        metrics = ev.evaluate()

        logging.info(f":Metrics ({str(metrics)}).")

        metrics["agg_metrics"] = metrics["map"]

        return metrics

    @dist_utils.main_process
    def report_metrics_reg(self, result_file):
        logging.info(f":Begin evaluation ({result_file}).")
        sys.stdout = None

        if self.eval_dataset_name == "vgcoco_reg":
            # prediction
            result = COCO(result_file)
            assert "predictions" in result.dataset
            cap_type = list(result.dataset['predictions'][0]['caption'].keys())[0]
            new_imgs = []
            for id, ann in result.anns.items():
                ann['caption'] = ann['caption'][cap_type][0]['caption']
                img = copy.deepcopy(result.imgs[ann["image_id"]])
                img["id"] = id
                ann["image_id"] = id
                new_imgs.append(img)
            result.dataset["images"] = new_imgs
            result.createIndex()

            # ground truth
            dataset_root = result.dataset['info']['dataset_root']
            annotation_root = result.dataset['info']['annotation_root']
            gt_file = os.path.join(dataset_root, annotation_root).replace("finetune_", "").replace("_captions", "")
            gt = COCO(gt_file)
            new_imgs = []
            for id, ann in gt.anns.items():
                ann['caption'] = ann['caption']['dense'][0]
                img = copy.deepcopy(gt.imgs[ann["image_id"]])
                img["id"] = id
                ann["image_id"] = id
                new_imgs.append(img)
            gt.dataset["images"] = new_imgs
            gt.createIndex()
        else:
            # prediction
            result = COCO(result_file)
            assert "predictions" in result.dataset
            cap_type = list(result.dataset['predictions'][0]['caption'].keys())[0]
            for id, ann in result.anns.items():
                ann['caption'] = ann['caption'][cap_type][0]['caption']
                # ann['caption'] = ann['caption']["dense"][0]

            # ground truth
            dataset_root = result.dataset['info']['dataset_root']
            annotation_root = result.dataset['info']['annotation_root']
            gt_file = os.path.join(dataset_root, annotation_root).replace("finetune_", "").replace("_captions", "")
            gt = COCO(gt_file)
            for id, ann in gt.anns.items():
                ann['caption'] = ann['caption']['dense'][0]

        # Create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(gt, result)

        # Evaluate results
        coco_eval.params['image_id'] = result.getImgIds()
        coco_eval.evaluate()

        metrics = copy.deepcopy(coco_eval.eval)
        metrics["METEOR"] = metrics["METEOR"] * 100
        metrics["CIDEr"] = metrics["CIDEr"] * 100

        sys.stdout = sys.__stdout__

        logging.info(f":Metrics ({str(metrics)}).")
        metrics["agg_metrics"] = metrics["METEOR"]
        return metrics


import logging
from omegaconf import OmegaConf
import torch.distributed as dist

from controlcap.datasets.dataset import ControlCapDataset

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]

    return cfg

@registry.register_builder("controlcap")
class ControlCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = ControlCapDataset
    eval_dataset_cls = ControlCapDataset

    def __init__(self, cfg=None):
        if isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                split=split,
                **dict(build_info),
                # caption_types=self.config.build_info.caption_types,
                # with_mask=self.config.build_info.with_mask,
                # mul=self.config.build_info.mul,
                # is_train=is_train,
            )

        return datasets
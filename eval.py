import argparse
import os.path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# imports modules for registration
from reem.tasks import *
from reem.datasets import *
from reem.models import *
from reem.runners import *
from reem.commom.config import Config

import lavis.tasks as tasks
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--res-path", required=True, help="path to the dense caption result")
    parser.add_argument("--metric", action="store_true", help="evaluate meteor score")
    parser.add_argument("--viz", action="store_true", help="visualize result")
    parser.add_argument("--local-rank", default=-1, type=int)  # for debug
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()
    task = tasks.setup_task(cfg)

    res_path = cfg.args.res_path
    assert os.path.exists(res_path)

    if cfg.args.metric:
        if ("reg" in task.eval_dataset_name) or ("refcoco" in task.eval_dataset_name):
            task.report_metrics_reg(res_path)
        else:
            task.report_metrics_densecap(res_path)

    if cfg.args.viz:
        task.visualize_result(res_path)


if __name__ == "__main__":
    main()

import importlib
import os

import torch
import logging
import numpy as np

from tqdm import tqdm
from thop import profile, clever_format
from torch.backends import cudnn

from Datasets.ModelNet40_Dataset import ModelNet
from Datasets.ShapeNet_Dataset import ShapeNet
from Models import build_model
from Utils.Config import get_args, get_config
from Utils.Tool import create_experiment_dir


def get_logger(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=log_dir + "/Perfo_Detec.txt", mode='a', encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt=fmt), sh.setFormatter(fmt=fmt)
    log.addHandler(fh), log.addHandler(sh)
    return log


def cal_flops_params(model, inputs):
    flops, _ = profile(model=model, inputs=inputs)
    flops = clever_format(flops, "%.3f")
    return flops


def cal_latency(model, inputs):
    repetitions = 300
    cudnn.benchmark = True
    # warmup
    print('warm up ...')
    with torch.no_grad():
        for _ in range(100):
            _ = model(*inputs)
    # statistics
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    print('testing ...')
    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = model(*inputs)
            ender.record()
            # waiting GPU
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    avg = timings.sum() / repetitions
    return avg


def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y,]


"""close all batch normalization layers in the ModelNet40 when testing due to only one sample is fed into the ModelNet40, else raising an error"""
if __name__ == "__main__":
    args = get_args()
    cfgs = get_config(args)

    cfgs.common.experiment_dir = os.path.join(cfgs.common.experiment_dir, cfgs.model.NAME, cfgs.dataset.NAME)
    create_experiment_dir(cfgs.common.experiment_dir)

    logger = get_logger(cfgs.common.experiment_dir)

    # prepare base model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model = build_model(cfgs.model).to(device)
    base_model.eval()

    # prepare inputs
    shapenet = ShapeNet(cfgs.dataset.test)
    num_point = 100000
    points, _, category = shapenet[0]
    points = torch.from_numpy(points)[None,].to(device)
    cat_prompt = to_categorical(category[None,], cfgs.model.category).to(device)
    inputs = (points, cat_prompt)

    # modelnet40 = ModelNet(cfgs.dataset.test)
    # _, _, (points, label) = modelnet40[0]
    # num_point = 1024
    # choice = np.random.choice(8192, num_point, replace=True)
    # points = points[choice]
    # points = torch.from_numpy(points)[None, ].to(device)
    # inputs = (points, )

    """calculate flops and params"""
    flops = cal_flops_params(base_model, inputs)

    """calculate latency"""
    avg_latency = cal_latency(base_model, inputs)

    logger.info("------- %s - %s" % (cfgs.common.experiment_dir, num_point))
    logger.info("flops: %s, avg_latency: %s  \n" % (flops, avg_latency))

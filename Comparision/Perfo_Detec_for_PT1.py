import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from thop import profile, clever_format
from torch.backends import cudnn
from Datasets.ModelNet40_Dataset import ModelNet
from Utils.Config import get_args, get_config

from model_pt import PointTransformerCls, PointTransformerSeg


def parse_args():
    """"PARAMETERS"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', default=13, type=int, choices=[15, 40], help='15: ScanObjectNN, 40: ModelNet40')
    parser.add_argument('--num_point', type=int, default=32768, help='Point Number')
    parser.add_argument('--nneighbor', type=int, default=52, help='pillar base size')
    parser.add_argument('--nblocks', type=int, default=4, help='pillar base size')
    parser.add_argument('--transformer_dim', type=int, default=512, help='pillar base size')
    parser.add_argument('--input_dim', type=int, default=3, help='pillar base size')
    return parser.parse_args()


def get_logger(log_dir):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=log_dir + "Perfo_Detec.txt", mode='a', encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt=fmt), sh.setFormatter(fmt=fmt)
    log.addHandler(fh), log.addHandler(sh)
    return log


def cal_flops_params(model, inputs):
    flops, params = profile(model=model, inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


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
    return torch.eye(num_classes)[y, ]


"""close all batch normalization layers in the ModelNet40 when testing due to only one sample is fed into the ModelNet40, else raising an error"""
if __name__ == "__main__":
    # cls_ModelNet, cls_ScanObjectNN, par_ShapeNet, sem_S3DIS
    obj = "cls_ModelNet"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if obj == "cls_ModelNet":
        """get ModelNet40"""
        model = PointTransformerCls(parse_args())
        model = model.to(device)
        model.eval()

        """prepare inputs"""
        args = get_args()
        cfgs = get_config(args)
        modelnet = ModelNet(cfgs.dataset.test)
        _, _, (points, _) = modelnet[0]
        num_point = 32768
        choice = np.random.choice(8192, num_point, replace=True)
        points = points[choice][:, :3]
        points = torch.from_numpy(points)[None, ].to(device)
        inputs = (points,)

        """calculate flops and params"""
        flops, params = cal_flops_params(model, inputs)

        """calculate latency"""
        # avg_latency = cal_latency(model, inputs)

        print("PT1 - %s - flops: %s, params: %s \n" % (obj, flops, params))

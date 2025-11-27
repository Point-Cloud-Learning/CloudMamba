import os
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset
from .Build_Dataset import datasets


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label


@datasets.register_module()
class S3DIS(Dataset):
    def __init__(self, cfgs_dataset_type):
        data_dir = cfgs_dataset_type.DATA_PATH
        test_area = "Area_5"
        self.train = True if cfgs_dataset_type.mode == 'train' else False
        all_files = _get_data_files(os.path.join(data_dir, "all_files.txt"))
        room_filelist = _get_data_files(os.path.join(data_dir, "room_filelist.txt"))

        data_batch_list, label_batch_list = [], []
        for f in all_files:
            data, label = _load_data_file(os.path.join(data_dir, f))
            data_batch_list.append(data)
            label_batch_list.append(label)

        data_batches = np.concatenate(data_batch_list, 0)
        labels_batches = np.concatenate(label_batch_list, 0)

        train_idx, test_idx = [], []
        for i, room_name in enumerate(room_filelist):
            if test_area in room_name:
                test_idx.append(i)
            else:
                train_idx.append(i)

        if self.train:
            self.points = data_batches[train_idx, ...]
            self.labels = labels_batches[train_idx, ...]
        else:
            self.points = data_batches[test_idx, ...]
            self.labels = labels_batches[test_idx, ...]

        # label weights
        labels = torch.tensor(self.labels).view(-1)
        self.num_labels = torch.bincount(labels, minlength=13)
        self.label_weights = torch.pow(torch.max(self.num_labels) / self.num_labels, 1 / 3.0)

    def __getitem__(self, item):
        return self.points[item], self.labels[item]

    def __len__(self):
        return 50

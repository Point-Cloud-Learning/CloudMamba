import os
import json
import pickle

import numpy as np
from tqdm import tqdm

from Utils.Logger import print_log
from .Build_Dataset import datasets
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)


@datasets.register_module()
class ShapeNet(Dataset):
    def __init__(self, cfgs_dataset_type):
        super(ShapeNet, self).__init__()
        self.root = cfgs_dataset_type.DATA_PATH
        self.num_point = cfgs_dataset_type.N_POINTS
        self.use_normals = cfgs_dataset_type.USE_NORMALS
        self.sampling = cfgs_dataset_type.UNIFORM_SAMPLING
        split = cfgs_dataset_type.mode

        self.cat_file = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        with open(self.cat_file, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'train':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.data_path = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.data_path.append((item, fn))

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.cache = {}

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, item):
        if item in self.cache:
            point_set, seg, cls = self.cache[item]
        else:
            fn = self.data_path[item]
            cat = fn[0]
            cls = self.classes[cat]
            cls = np.array(cls).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            seg = data[:, -1].astype(np.int32)
            if self.use_normals:
                point_set = data[:, 0:6]
            else:
                point_set = data[:, 0:3]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            self.cache[item] = (point_set, seg, cls)

        if self.sampling == "fps":
            centroids = farthest_point(point_set, self.num_point)
            point_set = point_set[centroids, :]
            seg = seg[centroids]
        elif self.sampling == "random":
            # replace=False means non-repetitive selection
            choice = np.random.choice(len(seg), self.num_point, replace=False)
            point_set = point_set[choice, :]
            seg = seg[choice]

        return point_set, seg, cls


# @datasets.register_module()
# class ShapeNet(Dataset):
#     def __init__(self, cfgs_dataset_type):
#         super(ShapeNet, self).__init__()
#         self.root = cfgs_dataset_type.DATA_PATH
#         self.num_point = cfgs_dataset_type.N_POINTS
#         self.use_normals = cfgs_dataset_type.USE_NORMALS
#         self.sampling = cfgs_dataset_type.UNIFORM_SAMPLING
#         split = cfgs_dataset_type.mode
#
#         self.cat_file = os.path.join(self.root, 'synsetoffset2category.txt')
#         self.cat = {}
#         with open(self.cat_file, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))
#
#         self.meta = {}
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
#             train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
#             test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         for item in self.cat:
#             self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item])
#             fns = sorted(os.listdir(dir_point))
#             if split == 'train':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn[0:-4] in test_ids]
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)
#
#             for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 self.meta[item].append(os.path.join(dir_point, token + '.txt'))
#
#         self.data_path = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.data_path.append((item, fn))
#
#         self.save_path = os.path.join(self.root, 'shapenet_%s_2048pts_fps.dat' % split)
#
#         if not os.path.exists(self.save_path):
#             print_log('Processing data %s (only running in the first time)...' % self.save_path, logger='ShapeNet')
#             self.list_of_points = [None] * len(self.data_path)
#             self.list_of_seg = [None] * len(self.data_path)
#             self.list_of_cls = [None] * len(self.data_path)
#
#             for index in tqdm(range(len(self.data_path)), total=len(self.data_path)):
#                 fn = self.data_path[index]
#                 cls = self.classes[fn[0]]
#                 cls = np.array([cls]).astype(np.int32)
#                 data = np.loadtxt(fn[1]).astype(np.float32)
#                 point_set = data[:, 0:-1]
#                 seg = data[:, -1].astype(np.int32)
#
#                 centroids = farthest_point(point_set, 2048)
#                 point_set = point_set[centroids, :]
#                 seg = seg[centroids]
#
#                 self.list_of_points[index], self.list_of_seg[index], self.list_of_cls[index] = point_set, seg, cls
#
#             with open(self.save_path, 'wb') as f:
#                 pickle.dump([self.list_of_points, self.list_of_seg, self.list_of_cls], f)
#         else:
#             print_log('Load processed data from %s...' % self.save_path, logger='ShapeNet')
#             with open(self.save_path, 'rb') as f:
#                 self.list_of_points, self.list_of_seg, self.list_of_cls = pickle.load(f)
#
#         self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                             'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
#                             'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
#                             'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
#                             'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
#
#     def __len__(self):
#         return len(self.data_path)
#
#     def __getitem__(self, item):
#         point_set, seg, cls = self.list_of_points[item], self.list_of_seg[item], self.list_of_cls[item]
#
#         if self.use_normals:
#             point_set = point_set[:, 0:6]
#         else:
#             point_set = point_set[:, 0:3]
#
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#         if self.sampling == "fps":
#             centroids = farthest_point(point_set, self.num_point)
#             point_set = point_set[centroids, :]
#             seg = seg[centroids]
#         elif self.sampling == "random":
#             # replace=False means non-repetitive selection
#             choice = np.random.choice(len(seg), self.num_point, replace=False)
#             point_set = point_set[choice, :]
#             seg = seg[choice]
#
#         return point_set, seg, cls



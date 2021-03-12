# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 17:21
@function:  实现原文作者的数据加载dataloader
"""

import numpy as np
import cv2
import os
import random
from torch.utils.data import DataLoader, Dataset
import torch
import time as t
from net.ouy_net import Network
from termcolor import cprint
from random import choice
import torchvision.transforms as transforms


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def np_to_variable(x, is_cuda=True, is_training=False, is_lable=False, dtype=torch.FloatTensor):
    if is_training:
        v = torch.from_numpy(x).type(dtype)
    elif is_lable:
        v = torch.tensor(x).type(dtype)

    # device = torch.device('cuda:1')
    # if torch.cuda.is_available():
    #     v = v.to(device)
    return v


# concat


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_path, data_name, transform=False, shuffle=True):
        # self.data_tensor = data_tensor
        # self.target_tensor = target_tensor
        self.data_path = data_path
        self.data_name = data_name
        # self.pre_load = pre_load
        # self.imresize = imresize;
        # self.sample=sample
        # self.samplenumber=samplenumber
        self.data_files = []
        self.label = []
        self.weights = []

        filename = os.path.join(data_path, data_name)
        with open(filename, 'r') as file_to_read:
            idx = 0
            name = []
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                    pass
                nameT, labelT = lines.split()
                nameT = os.path.splitext(nameT)[0] + "_RR" + os.path.splitext(nameT)[1]
                name.append(nameT)  # [0001_img.tiff,0002_img.tiff....]
                path = os.path.join(data_path, nameT)
                if os.path.isfile(path):
                    self.data_files.append(path)
                    self.label.append(labelT)  # [2,3,....]
        self.num_samples = len(self.data_files)

    def __getitem__(self, index):

        Set = set([0, 1, 2, 3])
        randN = int(choice(list(Set)))

        imgpath = self.data_files[index]
        target = self.label[index]
        target = np_to_variable(int(target), is_cuda=True, is_lable=True, dtype=torch.LongTensor)

        rgb = cv2.imread(imgpath, cv2.IMREAD_COLOR)

        rgb = np.copy(rgb)
        rgb = rgb.astype(np.float16, copy=False)
        rgb = rgb.transpose(2, 0, 1)  # shape(128,128,3)--- shape（3,128,128）
        rgb = rgb.reshape((rgb.shape[0], rgb.shape[1], rgb.shape[2]))  # shape（1,3,128,128）
        rgb = np_to_variable(rgb, is_cuda=True, is_training=True)

        hillshadepath = imgpath.replace('RR.jpg', 'S.txt')
        with open(hillshadepath, 'r') as file_to_read:
            lines = file_to_read.readline()
            nc = int(lines)
            lines = file_to_read.readline()
            nr = int(lines)
            shade = np.zeros((nr, nc))
            for i in range(0, nr):
                lines = file_to_read.readline()
                lines = lines.split()
                for j in range(0, nc):
                    shade[i, j] = float(lines[j])

        shade = np.copy(shade)
        shade = shade.reshape((1, shade.shape[0], shade.shape[1]))  # shape（1,1,128,128）
        shade = np_to_variable(shade, is_cuda=True, is_training=True)

        dem = []
        dempath = imgpath.replace('RR.jpg', 'D.txt')
        with open(dempath, 'r') as file_to_read:
            lines = file_to_read.readline()
            nameT, nC = lines.split()
            lines = file_to_read.readline()
            nameT, nR = lines.split()
            nR = int(nR)
            nC = int(nC)
            dem = np.zeros((nR, nC))  # shape（128,128）
            for i in range(0, nR):
                lines = file_to_read.readline()
                lines = lines.split()
                for j in range(0, nC):
                    dem[i, j] = float(lines[j])

        dem = np.copy(dem)
        # dem = cv2.flip(dem, 0)  # 1水平翻转,0垂直翻转,-1水平垂直翻转
        dem = dem.reshape((1, dem.shape[0], dem.shape[1]))  # shape（1,1,128,128）
        dem = np_to_variable(dem, is_cuda=True, is_training=True)

        return rgb, shade, dem, target, imgpath

    def __len__(self):
        return self.num_samples

    def getWeight(self):
        # class_sample_count = [100, 100, 100]
        class_sample_count = [200, 200, 200, 200, 200, 200]  # dataset has 10 class-1 samples, 1 class-2 samples, etc.
        classweights = 1 / torch.Tensor(class_sample_count)
        for i in self.label:
            if int(i) == 0:
                self.weights.append(classweights[0])
            elif int(i) == 1:
                self.weights.append(classweights[1])
            elif int(i) == 2:
                self.weights.append(classweights[2])
            elif int(i) == 3:
                self.weights.append(classweights[3])
            elif int(i) == 4:
                self.weights.append(classweights[4])
            elif int(i) == 5:
                self.weights.append(classweights[5])

        return self.weights

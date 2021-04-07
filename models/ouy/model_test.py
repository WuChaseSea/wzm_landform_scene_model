# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/4/2 17:50
@function:  自己临时用的模型测试文件
"""

import torch
from torchsummary import summary
import cv2
import numpy as np

from wzm_improved_1 import WzmModel1
from wzm_improved_2 import WzmModel2
from ouy_vgg_model import VggModel

if __name__ == '__main__':
    x1 = cv2.imread(
        r"E:\MyProject\graduation_remote_sence\code\wzm_landform_scene_model\data\ouy_second\test\00001_img.tiff")
    x1 = cv2.resize(x1, (128, 128))  # 默认双线性插值
    x1 = x1.transpose(2, 0, 1)  # shape(128,128,3)--- shape（3,128,128）
    x1 = x1.reshape(1, x1.shape[0], x1.shape[1], x1.shape[2])  # shape（1,3,128,128）
    x2 = cv2.imread(
        r"E:\MyProject\graduation_remote_sence\code\wzm_landform_scene_model\data\ouy_second\test\00001_sha.tiff",
        cv2.IMREAD_GRAYSCALE)
    x2 = cv2.resize(x2, (128, 128))  # 默认双线性插值
    x2 = x2.reshape(1, x2.shape[0], x2.shape[1])
    # x2 = x2.transpose(2, 0, 1)  # shape(128,128,3)--- shape（3,128,128）
    x2 = x2.reshape(1, x2.shape[0], x2.shape[1], x2.shape[2])  # shape（1,3,128,128）
    dempath = r"E:\MyProject\graduation_remote_sence\code\wzm_landform_scene_model\data\ouy_second\test\00001_dem.txt"
    dem = []
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
    dem = dem.reshape((dem.shape[0], dem.shape[1], 1))
    dem = cv2.resize(dem, (128, 128))
    dem = np.copy(dem)
    dem = dem.reshape(1, 1, dem.shape[0], dem.shape[1])  # shape（1,1,128,128）
    x1, x2, x3 = torch.randn((8, 3, 128, 128)), torch.randn((8, 1, 128, 128)), torch.randn((8, 1, 128, 128))
    model = WzmModel2()
    # x1 = torch.Tensor(x1)
    # x2 = torch.Tensor(x2)
    # dem = torch.Tensor(dem)
    # print(x1.shape)
    # print(x2.shape)
    # print(dem.shape)
    out = model(x1, x2, x3)
    # print(model)

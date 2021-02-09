# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 17:17
@function:  实现原文作者的模型
"""

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

from models.spp_net import SpatialPyramidPooling2d


class TripleModels(nn.Module):

    # Multi-column CNN
    def __init__(self, num_classes=3, num_level=3, pool_type='max_pool', use_spp=False):
        super(TripleModels, self).__init__()
        self.features1 = nn.Sequential(
            # input_channels 输入通道，output_channels输出通道即卷积核个数
            # [16, 3, 128, 128]
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # [16, 64, 31, 31]
            nn.ReLU(inplace=True),  # [16, 64, 31, 31]
            nn.MaxPool2d(kernel_size=3, stride=2),  # torch.Size([16, 64, 15, 15])
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # [16, 192, 15, 15]
            nn.ReLU(inplace=True),  # [16, 192, 15, 15]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [16, 192, 7, 7]
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # [16, 384, 7, 7]
            nn.ReLU(inplace=True),  # [16, 384, 7, 7]
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # [16, 256, 7, 7]
            nn.ReLU(inplace=True),  # [16, 256, 7, 7]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # [16, 256, 7, 7]
            nn.ReLU(inplace=True),  # [16, 256, 7, 7]
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # [16, 128, 7, 7]
            nn.ReLU(inplace=True),  # [16, 128, 7, 7]
            nn.MaxPool2d(kernel_size=3, stride=2),  # [16, 128, 3, 3]
        )
        self.features11 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv1_fusion = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1_fusion = nn.ReLU(inplace=True)

        self.conv2_fusion = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2_fusion = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        self.use_spp = use_spp
        self.num_level = num_level
        self.pool_type = pool_type
        self.num_grid = self._cal_num_grids(num_level)
        self.spp_layer = SpatialPyramidPooling2d(num_level)

        self.classifier_spp = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * self.num_grid, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def _cal_num_grids(self, level):
        count = 0
        for i in range(level):
            count += (i + 1) * (i + 1)
        return count

    def forward(self, x1, x2, x3, IsUseRGB):
        """
        :param x1: [16, 3, 128, 128]
        :param x2: [16, 1, 128, 128]
        :param x3: [16, 1, 128, 128]
        :param IsUseRGB:
        :return:
        """
        # x1---image ; x2-----dem ; x3 ----slope
        if IsUseRGB == 1:
            x1 = self.features1(x1)  # [16, 128, 3, 3]
        else:
            x1 = self.features11(x1)

        x2 = self.features2(x2)  # [16, 128, 3, 3]
        x3 = self.features3(x3)  # [16, 128, 3, 3]
        x = torch.cat((x1, x2, x3), 1)  # [16, 384, 3, 3]

        h = self.conv1_fusion(x)  # [16, 256, 3, 3]

        h = self.bn1(h)  # [16, 256, 3, 3]
        h = self.relu1_fusion(h)

        h = self.conv2_fusion(h)  # [16, 128, 3, 3]
        h = self.bn2(h)

        h += self.downsample(x)

        h = self.relu1_fusion(h)
        h = self.conv_fusion(h)  # [16, 64, 3, 3]
        # print(h.shape)
        if not self.use_spp:  # 如果use_spp为False
            h = h.view(h.size(0), -1)  # [16, 576]

            h = self.classifier(h)  # [16, 3]
        else:
            h = self.spp_layer(h)

            h = self.classifier_spp(h)
        return h

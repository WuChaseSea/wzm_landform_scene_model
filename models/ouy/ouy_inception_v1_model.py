# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/11 11:56
@function: 将原文作者的初始网络结构换成GoogLeNet网络提取特征图
"""

import torch.nn as nn
import torch

from models.spp_net import SpatialPyramidPooling2d


def ConvBNReLU(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                  padding=kernel_size // 2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


class InceptionV1Module(nn.Module):
    """
    单个inception模块
    """

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce,
                 out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()

        self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=5)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv2(self.branch3_conv1(x))
        out4 = self.branch4_conv1(self.branch4_pool(x))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionAux(nn.Module):
    """
    网络中单个输出softmax
    """

    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_linear1 = nn.Linear(in_features=128 * 2 * 2, out_features=1024)
        self.auxiliary_relu = nn.ReLU6(inplace=True)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.auxiliary_relu(self.auxiliary_linear1(x))
        out = self.auxiliary_linear2(self.auxiliary_dropout(x))
        return out


class InceptionV1(nn.Module):
    def __init__(self, num_classes=3, stage='train', num_level=3, pool_type='max_pool', use_spp=False):
        super(InceptionV1, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.block1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionV1Module(in_channels=192, out_channels1=64, out_channels2reduce=96, out_channels2=128,
                              out_channels3reduce=16, out_channels3=32, out_channels4=32),
            InceptionV1Module(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192,
                              out_channels3reduce=32, out_channels3=96, out_channels4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block4_1 = InceptionV1Module(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208,
                                          out_channels3reduce=16, out_channels3=48, out_channels4=64)

        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512, out_channels=num_classes)

        self.block4_2 = nn.Sequential(
            InceptionV1Module(in_channels=512, out_channels1=160, out_channels2reduce=112, out_channels2=224,
                              out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV1Module(in_channels=512, out_channels1=128, out_channels2reduce=128, out_channels2=256,
                              out_channels3reduce=24, out_channels3=64, out_channels4=64),
            InceptionV1Module(in_channels=512, out_channels1=112, out_channels2reduce=144, out_channels2=288,
                              out_channels3reduce=32, out_channels3=64, out_channels4=64),
        )

        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528, out_channels=num_classes)

        self.block4_3 = nn.Sequential(
            InceptionV1Module(in_channels=528, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                              out_channels3reduce=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block5 = nn.Sequential(
            InceptionV1Module(in_channels=832, out_channels1=256, out_channels2reduce=160, out_channels2=320,
                              out_channels3reduce=32, out_channels3=128, out_channels4=128),
            InceptionV1Module(in_channels=832, out_channels1=384, out_channels2reduce=192, out_channels2=384,
                              out_channels3reduce=48, out_channels3=128, out_channels4=128),
        )

        self.conv1_fusion = nn.Conv2d(3072, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1_fusion = nn.ReLU(inplace=True)

        self.conv2_fusion = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2_fusion = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(3072, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        # self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        # self.dropout = nn.Dropout(p=0.4)
        # self.linear = nn.Linear(in_features=1024, out_features=num_classes)

        self.use_spp = use_spp
        if use_spp:
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

    def forward_tmp(self, x, channels=3):
        if channels == 3:
            x = self.block1(x)
        elif channels == 1:
            x = self.block1_1(x)
        # x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux1 = x = self.block4_1(x)
        aux2 = x = self.block4_2(x)
        x = self.block4_3(x)
        out = self.block5(x)
        # out = self.avgpool(out)
        # out = self.dropout(out)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
        # if self.stage == 'train':
        #     aux1 = self.aux_logits1(aux1)
        #     aux2 = self.aux_logits2(aux2)
        #     return aux1, aux2, out
        # else:
        #     return out

    def forward(self, x1, x2, x3, IsUseRGB):
        """
        :param x1: [16, 3, 128, 128]
        :param x2: [16, 1, 128, 128]
        :param x3: [16, 1, 128, 128]
        :param IsUseRGB:
        :return:
        """
        # x1---image ; x2-----dem ; x3 ----slope
        # if IsUseRGB == 1:
        #     x1 = self.features1(x1)
        # else:
        #     x1 = self.features11(x1)
        x1 = self.forward_tmp(x1)  # [16, 1024, 4, 4]
        x2 = self.forward_tmp(x2, 1)
        x3 = self.forward_tmp(x3, 1)
        x = torch.cat((x1, x2, x3), 1)

        h = self.conv1_fusion(x)

        h = self.bn1(h)
        h = self.relu1_fusion(h)

        h = self.conv2_fusion(h)
        h = self.bn2(h)

        h += self.downsample(x)

        h = self.relu1_fusion(h)
        h = self.conv_fusion(h)
        # print(h.shape)
        if not self.use_spp:  # 如果use_spp为False
            h = h.view(h.size(0), -1)  # [16, 576]

            h = self.classifier(h)  # [16, 3]
        else:
            h = self.spp_layer(h)

            h = self.classifier_spp(h)
        return h

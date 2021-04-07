# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/3/31 20:24
@function:  采用类似fcn结构进行特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ouy.wzm_improved_model import FCN8s, FCN16s, FCN32s


class WzmModel1(nn.Module):
    def __init__(self, num_classes=3):
        super(WzmModel1, self).__init__()
        self.feature_rgb = FCN16s(n_class=num_classes)
        self.feature_shade = FCN8s(n_class=num_classes)
        self.feature_dem = FCN32s(n_class=num_classes)
        self.dem_to_rgb = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.shade_to_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        in_planes = 9
        planes = 16
        stride = 2
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
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

    def forward(self, x1, x2, x3, IsUseRGB=1):
        x1 = self.feature_rgb(x1)  # [1 3 8 8]
        x2 = self.feature_shade(x2)  # [1 3 16 16]
        x3 = self.feature_dem(x3)  # [1 3 4 4]
        print("x1 shape: ", x1.shape)
        print("x2 shape: ", x2.shape)
        print("x3 shape: ", x3.shape)
        x3 = self.dem_to_rgb(x3)  # [1 3 12 12]
        print("x3 shape: ", x3.shape)
        x2 = self.shade_to_rgb(x2)  # [1 3 12 12]
        print("x2 shape: ", x2.shape)
        x = torch.cat((x1, x2, x3[:, :, 1:9, 1:9]), 1)  # [1 9 12 12]
        print("x shape: ", x.shape)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        from ipdb import set_trace
        set_trace()

        print("out shape: ", out.shape)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        print("out shape: ", out.shape)
        return out


if __name__ == "__main__":
    wzm_model = WzmModel1()
    x1, x2, x3 = torch.randn((1, 3, 128, 128)), torch.randn((1, 1, 128, 128)), torch.randn((1, 1, 128, 128))
    # x1, x2, x3 = torch.randn((1, 3, 64, 64)), torch.randn((1, 1, 64, 64)), torch.randn((1, 1, 64, 64))
    wzm_model(x1, x2, x3)
    # print(out.shape)

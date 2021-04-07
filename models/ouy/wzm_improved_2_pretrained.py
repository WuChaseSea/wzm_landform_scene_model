# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/4/1 23:56
@function:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from models.se_module import SELayer


class WzmModel2(nn.Module):
    def __init__(self, num_classes=3):
        super(WzmModel2, self).__init__()
        self.fusion_upchannel = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.se = SELayer(16, reduction=4)
        self.conv_up = nn.Conv2d(16, 3, kernel_size=1)
        self.relu_up = nn.ReLU(inplace=True)
        self.conv_down1 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_down2 = nn.Conv2d(256, 64, kernel_size=1)
        self.relu_down = nn.ReLU(inplace=True)

        self.vgg = models.vgg16()
        del self.vgg.classifier
        origin_dicts = torch.load('pth/vgg16-397923af.pth')  # 预训练模型中vgg网络的参数
        model_dicts = self.vgg.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.vgg.load_state_dict(model_dicts)  # 加载更新后的自定义的网络模型参数

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
        x = torch.cat((x1, x2, x3), 1)
        x = self.fusion_upchannel(x)
        x = self.se(x)  # [1, 16, 128, 128]
        x = self.relu_up(self.conv_up(x))
        x = self.vgg.features(x)
        x = self.relu_down(self.conv_down1(x))
        x = self.relu_down(self.conv_down2(x))
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        print(x.shape)
        return x


if __name__ == "__main__":
    wzm_model = WzmModel2()
    x1, x2, x3 = torch.randn((1, 3, 128, 128)), torch.randn((1, 1, 128, 128)), torch.randn((1, 1, 128, 128))
    # x1, x2, x3 = torch.randn((1, 3, 64, 64)), torch.randn((1, 1, 64, 64)), torch.randn((1, 1, 64, 64))
    wzm_model(x1, x2, x3)
    # print(out.shape)

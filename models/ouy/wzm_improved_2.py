# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/4/1 23:56
@function:  对Triple模型的改进，优先进行融合，然后采用SE自适应调整通道重要性，再采用VGG完成特征提取；
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.se_module import SELayer

class WzmModel2(nn.Module):
    def __init__(self, num_classes=3):
        super(WzmModel2, self).__init__()
        self.fusion_upchannel = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.se = SELayer(16, reduction=4)
        self.conv_up = nn.Conv2d(3, 16, kernel_size=1)
        self.relu_up = nn.ReLU(inplace=True)
        self.features1 = nn.Sequential(
            # [16, 3, 128, 128]
            nn.Conv2d(16, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 4 * 4, 512),
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

        # x = self.fcn8s(x)
        x = self.features1(x)
        print(x.shape)
        # x = self.relu_up(self.conv_up(x))
        # x = self.conv7(x)
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

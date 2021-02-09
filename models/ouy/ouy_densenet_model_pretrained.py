# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/29 17:10
@function:
"""

import torch.nn as nn
import torch
import torchvision.models as models

from models.spp_net import SpatialPyramidPooling2d


class DenseNet121(nn.Module):
    def __init__(self, num_classes=3, num_level=3, pool_type='max_pool', use_spp=False):
        super(DenseNet121, self).__init__()
        self.densenet = models.densenet121(pretrained=False)
        del self.densenet.classifier
        # print(self.densenet)
        # print(self.densenet._modules.keys())
        origin_dicts = torch.load('pth/densenet121-a639ec97.pth')  # 预训练模型中densenet网络的参数
        model_dicts = self.densenet.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.densenet.load_state_dict(model_dicts)

        self.densenet_1 = models.densenet121(pretrained=False)
        del self.densenet_1.classifier
        self.densenet_1.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_dicts = self.densenet_1.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        layer1 = pretrained_dicts['features.conv0.weight']
        new = torch.zeros(64, 1, 7, 7)
        for i, output_channel in enumerate(layer1):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_dicts['features.conv0.weight'] = new
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.densenet_1.load_state_dict(model_dicts)

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

    def forward_3(self, x):
        out = self.densenet.features(x)
        return out

    def forward_1(self, x):
        out = self.densenet_1.features(x)
        return out

    def forward(self, x1, x2, x3, IsUseRGB=1):
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
        x1 = self.forward_3(x1)  # [16, 1024, 4, 4]
        x2 = self.forward_1(x2)
        x3 = self.forward_1(x3)
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


if __name__ == '__main__':
    dense = DenseNet121()
    out = dense(torch.randn(16, 3, 128, 128), torch.randn(16, 1, 128, 128), torch.randn(16, 1, 128, 128))
    print(out.shape)

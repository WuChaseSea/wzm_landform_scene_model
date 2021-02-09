# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/4 17:26
@function: 将原文作者的初始网络结构换成VGG网络提取特征图
"""

import torch.nn as nn
import torch
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from models.spp_net import SpatialPyramidPooling2d


class VggModel(nn.Module):
    def __init__(self, num_classes=3, num_level=3, pool_type='max_pool', use_spp=False):
        super(VggModel, self).__init__()
        self.vgg = models.vgg16()  # vgg16模型
        del self.vgg.avgpool
        del self.vgg.classifier  # 分别去掉已有vgg16的模型中的池化层和分类器
        self.vgg_1 = models.vgg16()  # 将输入通道改为1的vgg16模型
        del self.vgg_1.avgpool
        del self.vgg_1.classifier  # 分别去掉已有vgg16的模型中的池化层和分类器
        origin_dicts = torch.load('pth/vgg16-397923af.pth')  # 预训练模型中vgg网络的参数
        model_dicts = self.vgg.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.vgg.load_state_dict(model_dicts)  # 加载更新后的自定义的网络模型参数
        self.vgg_1.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 将输入通道改为1
        model_dicts = self.vgg_1.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        layer1 = pretrained_dicts['features.0.weight']
        new = torch.zeros(64, 1, 3, 3)
        for i, output_channel in enumerate(layer1):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_dicts['features.0.weight'] = new
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        # print(model_dicts.keys())
        self.vgg_1.load_state_dict(model_dicts)  # 加载更新后的自定义的网络模型参数
        # print(self.vgg_1.features)

        self.conv1_fusion = nn.Conv2d(1536, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1_fusion = nn.ReLU(inplace=True)

        self.conv2_fusion = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2_fusion = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(1536, 128, kernel_size=1, stride=1, padding=0, bias=False),
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

    def forward(self, x1, x2, x3, IsUseRGB=1):
        """
                :param x1: [16, 3, 128, 128]
                :param x2: [16, 3, 128, 128]
                :param x3: [16, 3, 128, 128]
                :param IsUseRGB:
                :return:
                """
        # x1---image ; x2-----dem ; x3 ----slope
        # if IsUseRGB == 1:
        #     x1 = self.features1(x1)
        # else:
        #     x1 = self.features11(x1)
        x1 = self.vgg.features(x1)  # [16, 128, 4, 4]

        x2 = self.vgg_1.features(x2)
        x3 = self.vgg_1.features(x3)
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


# if __name__ == '__main__':
#     vgg = MyVgg()
#     ou = vgg(torch.randn(1, 3, 128, 128), torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128))
#     print(ou.shape)
#     # print(vgg)

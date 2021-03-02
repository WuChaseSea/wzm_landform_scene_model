# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/21 16:23
@function:
"""

import torch
import torch.nn as nn
import torchvision.models as models

from models.se_module import SELayer


class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
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
        self.vgg_se = nn.Sequential()
        for i, fea in enumerate(self.vgg.features):
            print(i, fea)
            self.vgg_se.add_module(str(i), fea)
            if i == 16:
                self.vgg_se.add_module('se1', SELayer(256))
            if i == 23:
                self.vgg_se.add_module('se2', SELayer(512))
            if i == 30:
                self.vgg_se.add_module('se3', SELayer(512))
        model_dicts_se = self.vgg_se.state_dict()
        pretrained_dicts_se = {k[9:]: v for k, v in origin_dicts.items() if k[9:] in model_dicts_se}
        model_dicts_se.update(pretrained_dicts_se)
        self.vgg_se.load_state_dict(model_dicts_se)
        print(self.vgg_se[0])

    def forward(self, x):
        return x


if __name__ == '__main__':
    vgg = Vgg()
    # x = torch.randn(4, 3, 64, 64)
    # out = vgg(x)
    # print(out.shape)

# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 17:19
@function:  实现原文作者的网络结构
"""

import torch
import torch.nn as nn
from models.zism.zism_triple_model_pretrained import TripleModels
from models.zism.zism_vgg_model_pretrained import VggModel
from models.zism.zism_inception_v1_model_pretrained import InceptionV1
from models.zism.zism_inception_v2_model_pretrained import InceptionV2
from models.zism.zism_inception_v3_model_pretrained import InceptionV3
from models.zism.zism_inception_v4_model_pretrained import InceptionV4
from models.zism.zism_resnet_model_pretrained import ResNet34
from models.zism.zism_densenet_model_pretrained import DenseNet121


class Network(nn.Module):
    def __init__(self, index):
        super(Network, self).__init__()
        self.index = index
        if index == 'Triple':
            self.Models = TripleModels()
            # print('Using TripleModels')
        elif index == 'VggNet':
            self.Models = VggModel()
            # print('Using VggModel')
        elif index == 'GoogLeNet':
            self.Models = InceptionV1()
            # print('Using GoogLeNetModel')
        elif index == 'InceptionV2':
            self.Models = InceptionV2()
            # print('Using InceptionV2')
        elif index == 'InceptionV3':
            self.Models = InceptionV3()
            # print('Using InceptionV3')
        elif index == 'InceptionV4':
            self.Models = InceptionV4()
            # print('Using InceptionV4')
        elif index == 'ResNet34':
            self.Models = ResNet34()
            # print('Using ResNet34Model')
        elif index == 'DenseNet121':
            self.Models = DenseNet121()
            # print('Using DenseNet121')
        self.softmax = nn.LogSoftmax()
        self.loss_softmax = to_cuda(nn.CrossEntropyLoss())

    @property
    def loss(self):
        return self.loss_label

    def forward(self, im_data, dem_data, img_data, index, gt_data=None):

        im_data = to_cuda(im_data)
        dem_data = to_cuda(dem_data)
        # slope_data     = to_cuda(slope_data)
        img_data = to_cuda(img_data)

        pre_label = None
        # if self.index == 'Triple':
        #     pre_label = self.Models
        # elif self.index == 'VggNet':
        #     pre_label = self.Models
        # elif self.index == 'GoogLeNet':
        #     pre_label = self.Models
        # elif self.index == 'ResNet34':
        #     pre_label = self.Models(im_data, dem_data, img_data, 1)
        pre_label = self.Models(im_data, dem_data, img_data, 1)

        gt_data = to_cuda(gt_data)
        gt_data = gt_data.view(-1)
        self.loss_label = self.build_loss(pre_label, gt_data)  # åˆ°builds
        return pre_label

    def build_loss(self, pre_label, gt_label):
        loss = self.loss_softmax(pre_label, gt_label)
        return loss


def to_cuda(v):
    device = torch.device('cuda:0')
    if torch.cuda.is_available():
        v = v.to(device)
    return v

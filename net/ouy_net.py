# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 17:19
@function:  实现原文作者的网络结构
"""

import torch
import torch.nn as nn
from models.ouy.ouy_triple_model import TripleModels
from models.ouy.ouy_vgg_model import VggModel
from models.ouy.ouy_inception_v1_model import InceptionV1
from models.ouy.ouy_inception_v2_model import InceptionV2
from models.ouy.ouy_inception_v3_model import InceptionV3
from models.ouy.ouy_inception_v4_model import InceptionV4
from models.ouy.ouy_resnet_model import ResNet34
from models.ouy.ouy_densenet_model import DenseNet121



class Network(nn.Module):
    def __init__(self, index, use_spp=False, use_se=False):
        super(Network, self).__init__()
        self.index = index
        if index == 'Triple':
            self.Models = TripleModels(use_spp=use_spp)
            # print('Using TripleModels')
        elif index == 'VggNet':
            self.Models = VggModel(use_spp=use_spp, use_se=use_se)
            # print('Using VggModel')
        elif index == 'GoogLeNet':
            self.Models = InceptionV1(use_spp=use_spp)
            # print('Using GoogLeNetModel')
        elif index == 'InceptionV2':
            self.Models = InceptionV2(use_spp=use_spp)
            # print('Using InceptionV2')
        elif index == 'InceptionV3':
            self.Models = InceptionV3(use_spp=use_spp)
            # print('Using InceptionV3')
        elif index == 'InceptionV4':
            self.Models = InceptionV4(use_spp=use_spp)
            # print('Using InceptionV4')
        elif index == 'ResNet34':
            self.Models = ResNet34(use_spp=use_spp)
            # print('Using ResNet34Model')
        elif index == 'DenseNet121':
            self.Models = DenseNet121(use_spp=use_spp)
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

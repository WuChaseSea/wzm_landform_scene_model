# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/17 17:44
@function: 将原文作者的初始网络结构换成inception v3网络提取特征图
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class InceptionV3(nn.Module):
    def __init__(self, num_classes=3):
        super(InceptionV3, self).__init__()
        self.inception = models.Inception3(num_classes=3, aux_logits=False)
        del self.inception.fc
        # print(self.inception._modules.keys())
        # print(self.inception)
        origin_dicts = torch.load('pth/inception_v3_google-1a9a5a14.pth')  # 预训练模型中inception_v3网络的参数
        model_dicts = self.inception.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.inception.load_state_dict(model_dicts)
        self.inception_1 = models.Inception3(num_classes=3, aux_logits=False)
        del self.inception_1.fc
        self.inception_1.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        model_dicts = self.inception_1.state_dict()  # 自定义的去掉后面几层的网络的参数列表
        pretrained_dicts = {k: v for k, v in origin_dicts.items() if k in model_dicts}  # 预训练模型参数在自定义模型中有的参数列表
        layer1 = pretrained_dicts['Conv2d_1a_3x3.conv.weight']
        new = torch.zeros(32, 1, 3, 3)
        for i, output_channel in enumerate(layer1):
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrained_dicts['Conv2d_1a_3x3.conv.weight'] = new
        model_dicts.update(pretrained_dicts)  # 更新自定义的模型参数
        self.inception_1.load_state_dict(model_dicts)

        self.conv1_fusion = nn.Conv2d(6144, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1_fusion = nn.ReLU(inplace=True)

        self.conv2_fusion = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2_fusion = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(6144, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward_3(self, x):
        # N x 3 x 299 x 299
        x = self.inception.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.inception.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.inception.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        # x = F.dropout(x, training=True)
        # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # N x 2048
        # N x 1000 (num_classes)
        return x

    def forward_1(self, x):
        # N x 3 x 299 x 299
        x = self.inception_1.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_1.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_1.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.inception_1.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_1.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.inception_1.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_1.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_1.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_1.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_1.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_1.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_1.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_1.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_1.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_1.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_1.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        # x = F.dropout(x, training=True)
        # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # N x 2048
        # N x 1000 (num_classes)
        return x

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
        x1 = self.forward_3(x1)  # [16, 1024, 2, 2]
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
        h = h.view(h.size(0), -1)

        h = self.classifier(h)
        return h


# if __name__ == '__main__':
#     inc = InceptionV3()
#     out = inc(torch.randn(16, 3, 128, 128), torch.randn(16, 1, 128, 128), torch.randn(16, 1, 128, 128))
#     print(out.shape)
#     x = models.Inception3()
#     x1 = models.inception_v3(pretrained=False)
#     print('end')

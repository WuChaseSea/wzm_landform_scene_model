# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/29 15:44
@function:
"""
import torch
import torch.nn as nn
import torchvision

# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)

__all__ = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet264']


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class _TransitionLayer(nn.Module):
    def __init__(self, inplace, plance):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=plance, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(inplace),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16], num_classes=3):
        super(DenseNet, self).__init__()
        bn_size = 4
        drop_rate = 0
        self.conv1 = Conv1(in_planes=3, places=init_channels)
        self.conv1_1 = Conv1(in_planes=1, places=init_channels)

        num_features = init_channels
        self.layer1 = DenseBlock(num_layers=blocks[0], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        self.transition1 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer2 = DenseBlock(num_layers=blocks[1], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        self.transition2 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer3 = DenseBlock(num_layers=blocks[2], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        self.transition3 = _TransitionLayer(inplace=num_features, plance=num_features // 2)
        num_features = num_features // 2
        self.layer4 = DenseBlock(num_layers=blocks[3], inplances=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward_tmp(self, x, channels=3):
        if channels==3:
            x = self.conv1(x)
        else:
            x = self.conv1_1(x)

        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=3):
        super(DenseNet121, self).__init__()
        self.net = DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16])

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
            nn.Linear(64 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

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
        x1 = self.net.forward_tmp(x1)  # [16, 1024, 4, 4]
        x2 = self.net.forward_tmp(x2, 1)  # [16, 1024, 4, 4]
        x3 = self.net.forward_tmp(x3, 1)  # [16, 1024, 4, 4]
        x = torch.cat((x1, x2, x3), 1)  # [16, 3072, 4, 4]

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



def DenseNet169():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32])


def DenseNet201():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 48, 32])


def DenseNet264():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 64, 48])


if __name__ == '__main__':
    # model = torchvision.models.densenet121()
    model = DenseNet121()
    print(model)

    # out = model(torch.randn(8, 3, 128, 128),torch.randn(8, 1, 128, 128), torch.randn(8, 1, 128, 128))
    out = model(torch.randn(8, 3, 64, 64), torch.randn(8, 1, 64, 64), torch.randn(8, 1, 64, 64))
    print(out.shape)

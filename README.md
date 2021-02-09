# wzm_landform_scene_model
## 简介

参考文献：

 DU L, YOU X, LI K, et al. Multi­modal deep learning for landform recognition[J]. ISPRS Journal of Photogrammetry and Remote Sensing, 2019, 158 : 63 – 75.

主要原理：

![net](https://github.com/lover-520/wzm_landform_scene_model/blob/main/images/net.png)

原文作者的网络结构包括三个部分，前一部分是对山体阴影数据、高程数据和坡度数据通过卷积神经网络提取特征图，中间部分是特征融合部分，后面则是分类器的实现部分。该项目在原文作者的基础上，实现了常用的特征提取网络在地貌解译上的应用，即将前面的特征提取部分采用VggNet、GoogLeNet、Inception V2、Inception V3、Inception V4和ResNet、DenseNet等经典结构进行替换，并在不同的数据集上进行实验。

## 数据集

数据集的制作有些麻烦，有的用的图像，有的用的txt格式的，dataloader都不一样；

目前包含的数据集格式有：

ouy_dataloader：原始图像tiff格式，高程数据txt格式，山体阴影数据tiff格式；该数据是实验室做的，不方便公开，可用下面的zism数据测试；

zism_dataloader：图像jpg格式，高程数据txt格式，山体阴影数据txt格式，该数据原文中作者已提供，该项目中使用的进行处理过；

## 项目介绍

项目中含有使用预训练模型和不使用预训练模型两种的，所谓的使用预训练模型是加载那些经典网络在ImageNet数据集上的训练参数然后在自己的数据集上接着训练，而不是从头初始化再开始训练。

目前在原文作者的基础上，已添加的模型有：

VggNet、GoogLeNet、InceptionV2、InceptionV3、InceptionV4、ResNet、DenseNet

支持预训练的模型只有VggNet、GoogLeNet、InceptionV3、ResNet、DenseNet

已经实现的网络都支持使用spp结构

在使用预训练模型时，需要自行从网络上下载pth文件，放到pth文件夹里（其中，googlenet文件较小可以上传到GitHub上，其他的几个pth文件较大上传到GitHub上比较麻烦还没实现，下面给了几个pth文件的下载地址）

VggNet: 

```
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
```

GoogLeNet: 

```
model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}
```

InceptionV3: 

```
model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}
```

ResNet: 

```
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
```

DenseNet:

```
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}
```



## 运行说明

python achieve.py  后面需要加参数

- -train_path  训练集所在路径
- -valid_path  验证集所在路径
- -train_txt  训练集标签文件（该txt文件放在训练集里面，并且只要txt的文件名），默认train.txt
- -valid_txt  验证集标签文件（该txt文件放在验证集里面，并且只要txt的文件名），默认valid.txt
- -dataloader  需要加载的数据集格式，目前有ouy_dataloader和zism_dataloader两种格式，默认是zism_dataloader
- -model 选择的模型文件，目前有Triple、VggNet、GoogLeNet、InceptionV2、InceptionV3、InceptionV4和ResNet34，默认ResNet34
- -pretrained 是否需要使用预训练模型
- -use_spp 是否使用spp结构，默认False不使用
- -train_batchsize  训练时批处理的数目，默认1
- -valid_batchsize  验证时批处理的数目，默认1
- -epoches 迭代数，默认100

## 测试

python model_test/model_test.py

-test_path  测试集所在的文件夹

-test_txt  测试集的标签文件（该txt文件放在验证集里面，并且只要txt的文件名），默认值test.txt

-model  选用的模型文件，默认值ResNet34

-best_name  使用的模型参数文件

-dataloader  加载的数据集格式

-test_batchsize  测试时批处理的数目，默认值8
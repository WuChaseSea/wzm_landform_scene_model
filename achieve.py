# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/3 17:32
@function:  总实现文件
"""

import global_models as gm
import test as testmodel
import os
import sys
import argparse

import torch
import torchvision

gm._init()
gm.set_value('Triple', 'Triple')
gm.set_value('VggNet', 'VggNet')
gm.set_value('GoogLeNet', 'GoogLeNet')
gm.set_value('InceptionV2', 'InceptionV2')
gm.set_value('InceptionV3', 'InceptionV3')
gm.set_value('InceptionV4', 'InceptionV4')
gm.set_value('ResNet34', 'ResNet34')
gm.set_value('DenseNet121', 'DenseNet121')

# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)

def return_args():
    args_list = sys.argv
    if len(args_list) <= 5:
        print("参数少了")
        return None
    return args_list[1:]


def get_args():
    parser = argparse.ArgumentParser(description="get args...")

    parser.add_argument("-train_path", help="train image folder", type=str)
    parser.add_argument("-valid_path", help="valid image folder", type=str)

    parser.add_argument("-train_txt", help="train txt file", type=str, default="train.txt")
    parser.add_argument("-valid_txt", help="valid txt file", type=str, default="valid.txt")

    parser.add_argument("-dataloader", help="data_loader loader setting", type=str, default="zism_dataloader")

    parser.add_argument("-model", help="model", type=str, default="ResNet34")
    parser.add_argument("-pretrained", help="whether use pretrained model", type=str, default='False')

    parser.add_argument("-use_spp", help="whether use spp", type=str, default='False')
    parser.add_argument("-use_se", help="whether use se", type=str, default='False')

    parser.add_argument("-cross_validation", help="whether use k cross validation", type=str, default='False')

    parser.add_argument("-train_batchsize", help="train batchsize", type=int, default=2)
    parser.add_argument("-valid_batchsize", help="valid batchsize", type=int, default=2)

    parser.add_argument("-epoches", help="epoches", type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    # train_data_path = r'E:\MyProject\graduation_remote_sence\datasets\alllabel'
    # valid_data_path = train_data_path
    # train_data_txt = 'train.txt'
    # valid_data_txt = 'valid.txt'
    # index = 61
    # if return_args():
    #     train_data_path, valid_data_path, train_data_txt, valid_data_txt, index = return_args()
    # else:
    #     sys.exit()
    args = get_args()
    pretrained = args.pretrained
    use_spp = args.use_spp
    use_se = args.use_spp
    cross_validation = args.cross_validation
    if pretrained == 'False':
        pretrained = False
    else:
        pretrained = True
    if use_spp == 'False':
        use_spp = False
    else:
        use_spp = True
    use_se = True if use_se == 'True' else False
    cross_validation = True if cross_validation == 'True' else False
    if cross_validation:
        from model_train.train_cross_validation import train
    else:
        from model_train.train import train
    best_model = train(
        train_data_path=args.train_path,
        train_data_txt=args.train_txt,
        valid_data_path=args.valid_path,
        valid_data_txt=args.valid_txt,
        dataloader=args.dataloader,
        index=args.model,
        pretrained=pretrained,
        use_spp=use_spp,
        use_se=use_se,
        train_batch_size=args.train_batchsize,
        valid_batch_size=args.valid_batchsize,
        epoches=args.epoches,
    )
# best_modelName = method.train(index, tain_name, vali_name)  # 模型训练
# testmodel.test(index, best_modelName, vali_name)  # 使用验证集测试模型

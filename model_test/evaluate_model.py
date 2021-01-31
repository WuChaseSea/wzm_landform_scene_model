# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 17:52
@function:  测试模型精度
"""

from net.ouy_net import Network
import numpy as np
import torch
import os


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def evaluate_model(trained_model, data_loader, index):
    net = Network(index)
    load_net(trained_model, net)

    device = torch.device('cuda:0')
    if torch.cuda.is_available():
        net = net.to(device)
    net.eval()
    count = 0
    total = 0

    lableresultpath = trained_model.replace(".h5", ".txt")
    if os.path.exists(lableresultpath):
        os.remove(lableresultpath)
    valid_loss = 0.0

    for blob in data_loader:
        im_data = blob[0]
        dem_data = blob[2]
        img_data = blob[1]
        gt_data = blob[3].reshape((blob[3].shape[0], 1))
        index = 61
        pre_label = net(im_data, dem_data, img_data, index, gt_data)
        pre_label = pre_label.data.cpu().numpy()
        valid_loss += net.loss.item()

        label = pre_label.argmax(axis=1).flatten()
        num = len(label)

        for i in range(0, num):
            if gt_data[i] == label[i]:
                count = count + 1
            total = total + 1

    return 1.0 * count / total, valid_loss


def evaluate_model1(net, data_loader, index):

    device = torch.device('cuda:0')
    if torch.cuda.is_available():
        net = net.to(device)
    net.eval()
    count = 0
    total = 0

    # lableresultpath = trained_model.replace(".h5", ".txt")
    # if os.path.exists(lableresultpath):
    #     os.remove(lableresultpath)
    valid_loss = 0.0

    for blob in data_loader:
        im_data = blob[0]
        dem_data = blob[2]
        img_data = blob[1]
        gt_data = blob[3].reshape((blob[3].shape[0], 1))
        index = 61
        with torch.no_grad():
            pre_label = net(im_data, dem_data, img_data, index, gt_data)
        pre_label = pre_label.data.cpu().numpy()
        valid_loss += net.loss.item()

        label = pre_label.argmax(axis=1).flatten()
        num = len(label)

        for i in range(0, num):
            if gt_data[i] == label[i]:
                count = count + 1
            total = total + 1

    return 1.0 * count / total, valid_loss

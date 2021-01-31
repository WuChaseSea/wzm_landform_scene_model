# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/2 19:50
@function:
"""

import os
import sys

import numpy as np
import torch
import argparse
from model_timer import Timer

from net.ouy_net import Network
from evaluate_model import evaluate_model
import scipy.io as sio
import time
import global_models as gm


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def get_args():
    parser = argparse.ArgumentParser(description="get test args...")

    parser.add_argument("-test_path", help="test image folder", type=str)

    parser.add_argument("-test_txt", help="test txt file", type=str, default="test.txt")

    parser.add_argument("-model", help="model", type=str, default="ResNet34")

    parser.add_argument("-best_name", help="best name", type=str)

    parser.add_argument("-dataloader", help="dataloader", type=str)

    parser.add_argument("-test_batchsize", help="test batchsize", type=int, default=8)

    return parser.parse_args()


def model_test(nIndex, model_name, test_loader):
    # test_model.model_test(61, best_modelName, test_loader)

    path = "./model_save/model_" + nIndex + "_save/"
    model_path = path + model_name

    print('model_path', model_path)

    net = Network(nIndex)
    trained_model = os.path.join(model_path)
    load_net(trained_model, net)
    device = torch.device('cuda:0')
    if torch.cuda.is_available():
        net = net.to(device)
    net.eval()
    aprelable = []
    alable = []
    count = 0
    total = 0

    for blob in test_loader:
        im_data = blob[0]
        dem_data = blob[2]
        img_data = blob[1]
        gt_data = blob[3].reshape((blob[3].shape[0], 1))
        index = 61
        pre_label = net(im_data, dem_data, img_data, index, gt_data)
        pre_label = pre_label.data.cpu().numpy()

        label = pre_label.argmax(axis=1).flatten()
        num = len(label)
        for i in range(0, num):
            if gt_data[i] == label[i]:
                count = count + 1
            total = total + 1

            aprelable.append(label[i])
            alable.append(gt_data[i])

    # end=time.clock()
    # print (end-start)/300
    a = np.array(alable)
    b = np.array(aprelable)
    b = b.reshape(1, b.shape[0])
    a = a.reshape(1, a.shape[0])
    result = np.concatenate((a, b), axis=0);

    accu = int(10000.0 * count / total) / 100.0

    sName = '_accu(' + str(accu) + ').npy'

    model_path = model_path.replace('.h5', sName)

    # np.save(model_path, result)

    print('Accuracy:', accu)


if __name__ == '__main__':

    args = get_args()
    test_path = args.test_path
    test_name = args.test_txt
    nIndex = args.model
    best_name = args.best_name
    test_batchsize = args.test_batchsize
    dataloader = args.dataloader
    if dataloader == 'zism_dataloader':
        from data_loader.zism_dataloader import TensorDataset
    elif dataloader == 'ouy_dataloader':
        from data_loader.ouy_dataloader import TensorDataset
    test_dataset = TensorDataset(test_path, test_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batchsize, pin_memory=True,
                                              num_workers=8)

    model_test(nIndex, best_name, test_loader)

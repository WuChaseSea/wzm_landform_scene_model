# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/3 17:30
@function:  训练文件
"""

import os
import torch
import numpy as np
import random

from model_test.model_timer import Timer
from model_test.evaluate_model import evaluate_model, evaluate_model1
import global_models as gm


def get_allfold_data(txtpath, k):
    label0, label1, label2 = [], [], []  # 得到每一类的标签列表

    with open(txtpath, "r") as f:
        for line in f.readlines():
            picture, label = line.split()

            if label == "0":
                label0.append(picture)
            elif label == "1":
                label1.append(picture)
            elif label == "2":
                label2.append(picture)
    random.shuffle(label0)  # 将每一类的标签列表进行打乱
    random.shuffle(label1)
    random.shuffle(label2)

    labela, labelb, labelc, alllabel = [], [], [], []
    a, b, c, allpicture = [], [], [], []

    # divide 10-fold  10折交叉验证
    for i in range(k):
        a = label0[i * int(len(label0) / k):(i + 1) * int(len(label0) / k)]  # 第i折的第0类的数据
        b = label1[i * int(len(label1) / k):(i + 1) * int(len(label1) / k)]
        c = label2[i * int(len(label2) / k):(i + 1) * int(len(label2) / k)]
        labela = np.zeros((1, int(len(label0) / k)), dtype=np.uint8)
        labelb = np.ones((1, int(len(label1) / k)), dtype=np.uint8)
        labelc = np.ones((1, int(len(label2) / k)), dtype=np.uint8) * 2  # labelc中每一个元素都乘以2

        allpicture = allpicture + a + b + c  # 第i折的第0类、1类和2类的所有数据
        alllabel = alllabel + labela.tolist()[0] + labelb.tolist()[0] + labelc.tolist()[0]  # 第i折的标签分别为0、1、2
    return allpicture, alllabel  # 返回所有的图片名和标签


# print(len(allpicture),len(alllabel))

def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1  # 后面条件为True就正常执行
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = [], []
    for j in range(k):
        # idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        # print(idx)
        # idx 为每组 valid
        X_part, y_part = X[j * fold_size:(j + 1) * fold_size], y[j * fold_size:(j + 1) * fold_size]  # 第j折的数据，图片名列表和标签列表
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = X_train + X_part
            y_train = y_train + y_part
    c1 = list(zip(X_train, y_train))  # 将图像名列表与标签列表对应形成元组并将zip对象转换为list对象
    random.shuffle(c1)  # 打乱顺序
    X_train[:], y_train[:] = zip(*c1)  # 重新解压缩给X_train和y_train
    c2 = list(zip(X_valid, y_valid))  # 对需要验证的那一折数据也做同样处理
    random.shuffle(c2)
    X_valid[:], y_valid[:] = zip(*c2)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid  # 返回其他几折的训练集和某一折的验证集


def train(train_data_path, train_data_txt, valid_data_path, valid_data_txt, dataloader, index, pretrained,
          use_spp,
          train_batch_size,
          valid_batch_size, epoches, use_se=False, save_epoch=5, valid_epoch=5):
    # print(torch.cuda.get_device_name(0))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # if dataloader == 'zism_dataloader' and pretrained is False:
    #     from data_loader.zism_dataloader import TensorDataset, save_net
    #     from net.zism_net import Network
    # if dataloader == 'zism_dataloader' and pretrained is True:
    #     from data_loader.zism_dataloader import TensorDataset, save_net
    #     from net.zism_net_pretrained import Network
    # if dataloader == 'ouy_dataloader' and pretrained is False:
    #     from data_loader.ouy_dataloader import TensorDataset, save_net
    #     from net.ouy_net import Network
    # if dataloader == 'ouy_dataloader' and pretrained is True:
    #     from data_loader.ouy_dataloader import TensorDataset, save_net
    #     from net.ouy_net_pretrained import Network
    if dataloader == 'ouy_dataloader64' and pretrained is False:
        from data_loader.ouy_dataloader_64_cross_validation import TensorDataset, save_net
        from net.ouy_net import Network
    if dataloader == 'ouy_dataloader64' and pretrained is True:
        from data_loader.ouy_dataloader_64_cross_validation import TensorDataset, save_net
        from net.ouy_net_pretrained import Network

    k = 5
    allpicture,  alllabel = get_allfold_data(train_data_txt, k)
    
    train_loss_dict = {}
    valid_loss_dict = {}
    accuracy_dict = {}

    for ik in range(k):
        print('Using ' + gm.get_value(index) + '...')
        net = Network(index=index, use_spp=use_spp)
        device = torch.device('cuda:0')
        if torch.cuda.is_available():
            net = net.to(device)

        net.train()

        # training configuration
        start_step = 0
        end_step = epoches
        lr = 0.00001
        momentum = 0.9

        # ------------
        rand_seed = 64678
        if rand_seed is not None:
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)

        params = list(net.parameters())
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

        # training
        step_cnt = 0
        re_cnt = False
        t = Timer()
        t.tic()

        # dataCount = len(train_data_loader)

        best = 0

        best_model = ''

        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, ik, allpicture, alllabel)

        train_data_loader_tmp = TensorDataset(train_data_path, X_train, y_train)
        weights = train_data_loader_tmp.getWeight()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(X_train), replacement=True)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_data_loader_tmp, batch_size=train_batch_size,
                                                        pin_memory=True, sampler=sampler, num_workers=0)

        for epoch in range(start_step, end_step + 1):

            net.train()
            step = -1
            train_loss = 0
            for i, blob in enumerate(train_data_loader):
                step = step + 1
                im_data = blob[0]
                dem_data = blob[2]
                # slope     = blob['slope']
                img_data = blob[1]
                gt_data = blob[3].reshape((blob[3].shape[0], 1))
                # print(epoch, step)
                pre_label = net(im_data, dem_data, img_data, index, gt_data)
                loss = net.loss

                train_loss += loss.item()
                step_cnt += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log_text_per_loader = 'index: %s, epoch: %4d, cross_num: %4d, step: %4d, loss: %4.4f' % (
                    index, epoch, ik, i, loss.item()
                )
                print(log_text_per_loader)

            # 每运行一个epoch就打印出一次相关信息
            duration = t.toc(average=False)
            fps = step_cnt / duration
            log_text = 'index: %4s, epoch: %4d, all steps: %4d, Time: %.4fs, Duration: %.4fs, loss: %4.4f' % (
                index, epoch, step, 1. / fps, duration, train_loss)
            print(log_text)
            re_cnt = True
            epoch_str = str(ik*epoches + epoch)
            train_loss_dict[epoch_str] = train_loss

            if re_cnt:
                t.tic()
                re_cnt = False

            method, dataset_name = gm.get_value(index), dataloader.split('_')[0]

            if epoch % save_epoch == 0:  # 每5个epoch之后就保存一次模型
                output_dir = './model_save/model_%s_save' % index
                if not os.path.exists(output_dir):  # 如果目录不存在的话
                    # os.mkdir(output_dir)
                    os.makedirs(output_dir)
                if pretrained is True:
                    filename = '{}th_{}_{}_{}_pretrained.h5'.format(ik, dataset_name, method, epoch)
                else:
                    filename = '{}th_{}_{}_{}.h5'.format(ik, dataset_name, method, epoch)
                save_name = os.path.join(output_dir, filename)
                save_net(save_name, net)

            if epoch % valid_epoch == 0:  # 每5个epoch之后就保存一次模型并在验证集上测试一次
                valid_dataset = TensorDataset(train_data_path, X_valid, y_valid)
                valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, pin_memory=True, num_workers=6)

                # precision, valid_loss = evaluate_model(save_name, valid_data_loader, index)  # 100是传入这个模型的batchsize
                precision, valid_loss = evaluate_model1(net, valid_data_loader, index)
                valid_loss_dict[epoch_str] = valid_loss
                accuracy_dict[epoch_str] = precision
                if precision > best:
                    best = precision
                    if pretrained is True:
                        best_model = '{}th_{}_{}_{}_pretrained.h5'.format(ik, dataset_name, method, epoch)
                    else:
                        best_model = '{}th_{}_{}_{}.h5'.format(ik, dataset_name, method, epoch)
                # log_text = 'EPOCH: %d, load precision: %.4f, valid loss: %.4f, net precision: %.4f, net valid loss:
                # %.4f' % ( epoch, precision, valid_loss, precision1, valid_loss1) log_text = 'EPOCH: %d, precision:
                # %.4f, valid loss: %.4f' % (epoch, precision, valid_loss)
                log_text = 'EPOCH: %d, precision: %.4f' % (epoch, precision)
                # log_text = 'EPOCH: %d, precision: %.4f, valid loss: %.4f' % (epoch, precision, valid_loss)

                print(log_text)
                # log_print(log_text, color='green', attrs=['bold'])
                log_text = 'BEST Precision: %0.4f, BEST MODEL: %s' % (best, best_model)
                # log_print(log_text, color='green', attrs=['bold'])
                print(log_text)

            if epoch % 5 == 0:
                dataloader_tmp = dataloader.split('_')[0] + '_'
                accuracy_txt_output_dir = './model_save/model_%s_save' % index
                if pretrained is True:
                    accuracy_txt_output_name = dataloader_tmp + 'accuracy_pretrained.txt'
                else:
                    accuracy_txt_output_name = dataloader_tmp + 'accuracy.txt'
                accuracy_txt_output_file = os.path.join(accuracy_txt_output_dir, accuracy_txt_output_name)
                with open(accuracy_txt_output_file, 'w') as fw_train:
                    for epoch_key in accuracy_dict.keys():
                        fw_train.write(epoch_key)
                        fw_train.write('    ')
                        fw_train.write(str(accuracy_dict[epoch_key]))
                        fw_train.write('\n')
                print('finished write accuracy file tmp.')

    dataloader_tmp = dataloader.split('_')[0] + '_'

    train_txt_output_dir = './model_save/model_%s_save' % index
    if pretrained is True:
        train_txt_output_name = dataloader_tmp + 'train_loss_pretrained.txt'
    else:
        train_txt_output_name = dataloader_tmp + 'train_loss.txt'
    train_txt_output_file = os.path.join(train_txt_output_dir, train_txt_output_name)
    with open(train_txt_output_file, 'w') as fw_train:
        for epoch_key in train_loss_dict.keys():
            fw_train.write(epoch_key)
            fw_train.write('    ')
            fw_train.write(str(train_loss_dict[epoch_key]))
            fw_train.write('\n')
    print('finished write train loss file')

    valid_txt_output_dir = './model_save/model_%s_save' % index
    if pretrained is True:
        valid_txt_output_name = dataloader_tmp + 'valid_loss_pretrained.txt'
    else:
        valid_txt_output_name = dataloader_tmp + 'valid_loss.txt'
    valid_txt_output_file = os.path.join(valid_txt_output_dir, valid_txt_output_name)
    with open(valid_txt_output_file, 'w') as fw_train:
        for epoch_key in valid_loss_dict.keys():
            fw_train.write(epoch_key)
            fw_train.write('    ')
            fw_train.write(str(valid_loss_dict[epoch_key]))
            fw_train.write('\n')
    print('finished write valid loss file')

    accuracy_txt_output_dir = './model_save/model_%s_save' % index
    if pretrained is True:
        accuracy_txt_output_name = dataloader_tmp + 'accuracy_pretrained.txt'
    else:
        accuracy_txt_output_name = dataloader_tmp + 'accuracy.txt'
    accuracy_txt_output_file = os.path.join(accuracy_txt_output_dir, accuracy_txt_output_name)
    with open(accuracy_txt_output_file, 'w') as fw_train:
        for epoch_key in accuracy_dict.keys():
            fw_train.write(epoch_key)
            fw_train.write('    ')
            fw_train.write(str(accuracy_dict[epoch_key]))
            fw_train.write('\n')
    print('finished write accuracy file')
    # best_model_name = best_model
    # return best_model_name

# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/3 17:30
@function:  训练文件
"""

import os
import torch
import numpy as np

from model_test.model_timer import Timer
from model_test.evaluate_model import evaluate_model, evaluate_model1
import global_models as gm


def train(train_data_path, train_data_txt, valid_data_path, valid_data_txt, dataloader, index, pretrained,
          use_spp,
          train_batch_size,
          valid_batch_size, epoches, save_epoch=5, valid_epoch=2):
    # print(torch.cuda.get_device_name(0))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if dataloader == 'zism_dataloader' and pretrained is False:
        from data_loader.zism_dataloader import TensorDataset, save_net
        from net.zism_net import Network
    if dataloader == 'zism_dataloader' and pretrained is True:
        from data_loader.zism_dataloader import TensorDataset, save_net
        from net.zism_net_pretrained import Network
    if dataloader == 'ouy_dataloader' and pretrained is False:
        from data_loader.ouy_dataloader import TensorDataset, save_net
        from net.ouy_net import Network
    if dataloader == 'ouy_dataloader' and pretrained is True:
        from data_loader.ouy_dataloader import TensorDataset, save_net
        from net.ouy_net_pretrained import Network
    if dataloader == 'ouy_dataloader64' and pretrained is False:
        from data_loader.ouy_dataloader_64 import TensorDataset, save_net
        from net.ouy_net import Network
    if dataloader == 'ouy_dataloader64' and pretrained is True:
        from data_loader.ouy_dataloader_64 import TensorDataset, save_net
        from net.ouy_net_pretrained import Network

    train_loss_dict = {}
    valid_loss_dict = {}
    accuracy_dict = {}
    train_data_loader_tmp = TensorDataset(train_data_path, train_data_txt)
    weights = train_data_loader_tmp.getWeight()
    fw_train = open(os.path.join(train_data_path, train_data_txt), 'r')
    num_data_trains = len(fw_train.readlines())
    fw_train.close()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_data_trains, replacement=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_loader_tmp, batch_size=train_batch_size,
                                                    pin_memory=True, sampler=sampler, num_workers=0)

    valid_data_loader_tmp = TensorDataset(valid_data_path, valid_data_txt)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data_loader_tmp, batch_size=valid_batch_size,
                                                    pin_memory=True, num_workers=0)

    valid_data_loader_tmp = TensorDataset(valid_data_path, valid_data_txt)
    valid_data_loader = torch.utils.data.DataLoader(dataset=valid_data_loader_tmp, batch_size=valid_batch_size,
                                                    pin_memory=True, num_workers=0)

    # if index == 'Triple':
    #     print('Using TripleModel...')
    # elif index == 'VggNet':
    #     print('Using VggModel...')
    # elif index == 'GoogLeNet':
    #     print('Using GoogLeNet...')
    # elif index == 'ResNet34':
    #     print('Using ResNet34...')
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
            log_text_per_loader = 'index: %s, epoch: %4d, step: %4d, loss: %4.4f' % (
                index, epoch, i, loss.item()
            )
            print(log_text_per_loader)

        # 每运行一个epoch就打印出一次相关信息
        duration = t.toc(average=False)
        fps = step_cnt / duration
        log_text = 'index: %4s, epoch: %4d, all steps: %4d, Time: %.4fs, Duration: %.4fs, loss: %4.4f' % (
            index, epoch, step, 1. / fps, duration, train_loss)
        print(log_text)
        re_cnt = True
        epoch_str = str(epoch)
        train_loss_dict[epoch_str] = train_loss

        # if epoch % 1 == 0:
        #     duration = t.toc(average=False)
        #     fps = step_cnt / duration
        #     log_text = 'index: %4d, epoch: %4d, all steps: %4d, Time: %.4fs, Duration: %.4fs, loss: %4.4f' % (
        #         index, epoch, step, 1. / fps, duration, train_loss)
        #     print(log_text)
        #     # log_print(log_text, color='green', attrs=['bold'])
        #     re_cnt = True

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
                filename = '{}_{}_{}_pretrained.h5'.format(dataset_name, method, epoch)
            else:
                filename = '{}_{}_{}.h5'.format(dataset_name, method, epoch)
            save_name = os.path.join(output_dir, filename)
            save_net(save_name, net)

        if epoch % valid_epoch == 0:  # 每5个epoch之后就保存一次模型并在验证集上测试一次
            # method, dataset_name = gm.get_value(index), 'ouy'
            # if index == 'Triple':
            #     method = 'AlexNet'
            #     dataset_name = 'ouy'
            # elif index == 'VggNet':
            #     method = 'VggNet'
            #     dataset_name = 'ouy'
            # elif index == 'GoogLeNet':
            #     method = 'GoogLeNet_v1'
            #     dataset_name = 'ouy'
            # elif index == 'ResNet34':
            #     method = 'ResNet34'
            #     dataset_name = 'ouy'
            # output_dir = './model_save/model_%s_save' % index
            # if not os.path.exists(output_dir):  # 如果目录不存在的话
            #     # os.mkdir(output_dir)
            #     os.makedirs(output_dir)
            # save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
            # save_net(save_name, net)
            # calculate error on the validation dataset

            # precision, valid_loss = evaluate_model(save_name, valid_data_loader, index)  # 100是传入这个模型的batchsize
            precision, valid_loss = evaluate_model1(net, valid_data_loader, index)
            valid_loss_dict[epoch_str] = valid_loss
            accuracy_dict[epoch_str] = precision
            if precision > best:
                best = precision
                if pretrained is True:
                    best_model = '{}_{}_{}_pretrained.h5'.format(dataset_name, method, epoch)
                else:
                    best_model = '{}_{}_{}.h5'.format(dataset_name, method, epoch)
            # log_text = 'EPOCH: %d, load precision: %.4f, valid loss: %.4f, net precision: %.4f, net valid loss: %.4f' % (
            # epoch, precision, valid_loss, precision1, valid_loss1)
            # log_text = 'EPOCH: %d, precision: %.4f, valid loss: %.4f' % (epoch, precision, valid_loss)
            log_text = 'EPOCH: %d, precision: %.4f' % (epoch, precision)
            # log_text = 'EPOCH: %d, precision: %.4f, valid loss: %.4f' % (epoch, precision, valid_loss)

            print(log_text)
            # log_print(log_text, color='green', attrs=['bold'])
            log_text = 'BEST Precision: %0.4f, BEST MODEL: %s' % (best, best_model)
            # log_print(log_text, color='green', attrs=['bold'])
            print(log_text)

    dataloader_tmp = dataloader.split('_')[0] + '_'

    train_txt_output_dir = './model_save/model_%s_save' % index
    if pretrained is True:
        train_txt_output_name = dataloader_tmp + 'train_loss_pretrained.txt'
    else:
        train_txt_output_name = dataloader_tmp + 'train_loss.txt'
    train_txt_output_file  = os.path.join(train_txt_output_dir, train_txt_output_name)
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

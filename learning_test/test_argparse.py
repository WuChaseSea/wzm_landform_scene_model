# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/15 11:09
@function:  用来测试argparse的文件
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="get args...")

    parser.add_argument("-train_path", help="train image folder", type=str)
    parser.add_argument("-valid_path", help="valid image folder", type=str)
    parser.add_argument("-test_path", help="test image folder", type=str)

    parser.add_argument("-train_txt", help="train txt file", type=str, default="train.txt")
    parser.add_argument("-valid_txt", help="valid txt file", type=str, default="valid.txt")
    parser.add_argument("-test_txt", help="test txt file", type=str, default="test.txt")

    parser.add_argument("-model", help="model", type=str, default="ResNet34")

    parser.add_argument("-train_batchsize", help="train batchsize", type=int, default=8)
    parser.add_argument("-valid_batchsize", help="valid batchsize", type=int, default=8)
    parser.add_argument("-test_batchsize", help="test batchsize", type=int, default=8)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

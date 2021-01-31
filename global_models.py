# -*- coding: utf-8 -*-
"""
@author: WZM
@time: 2021/1/17 16:19
@function: 定义所有网络的文件
"""


def _init():
    global _global_models
    _global_models = {}  # 以网络名称做键和值


def set_value(name, value):
    _global_models[name] = value


def get_value(name, def_value=None):
    try:
        return _global_models[name]
    except KeyError:
        return def_value

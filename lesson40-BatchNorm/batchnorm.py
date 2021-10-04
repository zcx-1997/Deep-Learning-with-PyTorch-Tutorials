# -*- coding: utf-8 -*-

"""
@Time : 2021/10/3
@Author : Lenovo
@File : batchnorm
@Description : 
"""
import torch
from torch import nn

x = torch.randn(100, 16, 784)
layer = nn.BatchNorm1d(16)
out = layer(x)
print(x.mean(), x.std())
print(layer.running_mean)
print(layer.running_var)

x = torch.randn(1, 16, 7, 7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print(layer.running_mean)
print(layer.running_var)
print(vars(layer))  # Y和B 参数命名为 w 和 b
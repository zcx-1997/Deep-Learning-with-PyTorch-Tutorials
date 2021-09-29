# -*- coding: utf-8 -*-

"""
@Time : 2021/9/29
@Author : Lenovo
@File : crossEntrogy
@Description : 
"""
import torch

a = torch.full([4], 1 / 4)
print(a)

entropy = -(a * torch.log2(a)).sum()
print(entropy)

a = torch.tensor([0.1, 0.1, 0.1, 0.7])
entropy = -(a * torch.log2(a)).sum()
print(entropy)


a = torch.tensor([0.001, 0.001, 0.001, 0.999])
entropy = -(a * torch.log2(a)).sum()
print(entropy)
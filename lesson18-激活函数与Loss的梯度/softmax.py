# -*- coding: utf-8 -*-

"""
@Time : 2021/9/29
@Author : Lenovo
@File : softmax
@Description : 
"""
import torch
from torch.nn import functional as F

a = torch.rand(3).requires_grad_()
print("原始输出：data={}，sum={}".format(a, a.sum()))
# 原始输出：data=tensor([0.1932, 0.0932, 0.8357], requires_grad=True)，sum=1.1220511198043823

p = F.softmax(a, dim=0)
print("softmax输出：data={}，sum={}".format(p, p.sum()))
# softmax输出：data=tensor([0.2627, 0.2377, 0.4995], grad_fn=<SoftmaxBackward>)，sum=1.0

p1 = F.sigmoid(a)
print("sigmoid输出：data={}，sum={}".format(p1, p1.sum()))
# sigmoid输出：data=tensor([0.5481, 0.5233, 0.6976], grad_fn=<SigmoidBackward>)，sum=1.7689813375473022

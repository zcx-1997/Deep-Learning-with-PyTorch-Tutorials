# -*- coding: utf-8 -*-

"""
@Time : 2021/9/29
@Author : Lenovo
@File : crossEntrogy
@Description : 
"""
import torch
import torch.nn.functional as F

# entropy

a = torch.full([4], 1 / 4)
print(a)

entropy = -(a * torch.log2(a)).sum()
print(entropy)  # tensor(2.)

a = torch.tensor([0.1, 0.1, 0.1, 0.7])
entropy = -(a * torch.log2(a)).sum()
print(entropy)  # tensor(1.3568)

a = torch.tensor([0.001, 0.001, 0.001, 0.999])
entropy = -(a * torch.log2(a)).sum()
print(entropy)  #tensor(0.0313)


# cross entropy

x = torch.randn(1, 784)
w = torch.randn(784, 10)
label = torch.tensor([3])

logits = x @ w
print(logits.shape)

ce1 = F.cross_entropy(logits, label)
print("F.crossentropy输出:", ce1)
#F.crossentropy输出: tensor(69.6985)

# pytorch中：crossentropy = softmax+log+nll_loss

pred = F.softmax(logits, dim=1)
print(pred.shape)

pred_log = torch.log(pred)
ce2 = F.nll_loss(pred_log, label)
print("softmax+log+nll_loss输出:", ce2)
# softmax+log+nll_loss输出: tensor(69.6985)

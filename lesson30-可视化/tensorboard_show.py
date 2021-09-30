# -*- coding: utf-8 -*-

"""
@Time : 2021/9/30
@Author : Lenovo
@File : tensorboard_show
@Description : 
"""

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
writer.close()

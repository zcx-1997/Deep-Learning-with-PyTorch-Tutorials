# -*- coding: utf-8 -*-

"""
@Time : 2021/9/30
@Author : Lenovo
@File : visdom_show
@Description : 
"""
from visdom import Visdom
import numpy as np
import time

# 实例化窗口
wind = Visdom()
# 初始化窗口参数
wind.line([[0.,0.]],  # 纵轴初始点坐标
          [0.],       # 横轴初始点坐标
          win = 'train',  #窗口名称
          opts = dict(title = 'loss&acc',legend = ['loss','acc']))  # 图例

# 更新窗口数据
for step in range(10):
	loss = 0.2 * np.random.randn() + 1
	acc = 0.1 * np.random.randn() + 0.5
	wind.line([[loss, acc]],[step],win = 'train',update = 'append')
	time.sleep(0.5)

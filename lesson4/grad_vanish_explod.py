# -*- coding: utf-8 -*-
"""

梯度消失与爆炸
"""
import torch
import torch.nn as nn
from common_tools import set_seed
import numpy as np

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, unit_nums, layer_nums):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(unit_nums, unit_nums, bias=False) for i in range(layer_nums)]
        )
        self.unit_nums = unit_nums

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # x = torch.nn.Tanh()(x)
            x = torch.nn.ReLU()(x)

            print("layer:{}, std:{}".format(i, x.std()))
            if torch.isnan(x.std()):
                print("output is nan in layer {}!".format(i))
                break
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                # ================= normal initialization =================
                # nn.init.normal_(m.weight.data)  # normal: mean=0, std=1
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.unit_nums))

                # ================= xavier initialization =================
                # a = np.sqrt(6 / (self.unit_nums + self.unit_nums))
                # tanh_gain = nn.init.calculate_gain('tanh')  # 计算tanh激活函数的增益值
                # a *= tanh_gain
                # nn.init.uniform_(m.weight.data, -a, a)

                # tanh_gain = nn.init.calculate_gain('tanh')  # 计算tanh激活函数的增益值
                # nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

                # ================= kaiming initialization =================
                # nn.init.normal_(m.weight.data, std=np.sqrt(2 / self.unit_nums))
                nn.init.kaiming_normal_(m.weight.data)


flag = True
# flag = False

if flag:
    layer_nums = 100
    unit_nums = 256
    batch_size = 16

    net = MLP(unit_nums, layer_nums)
    net.initialize()

    inputs = torch.randn((batch_size, unit_nums))  # normal: mean=0, std=1

    output = net(inputs)
    print(output)

# ======================================= calculate gain =======================================

# flag = True
flag = False

if flag:
    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print("gain:{}".format(gain))

    tanh_gain = nn.init.calculate_gain("tanh")
    print("tanh_gain in PyTorch:", tanh_gain)

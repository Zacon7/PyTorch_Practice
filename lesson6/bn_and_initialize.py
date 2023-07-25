# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from common_tools import set_seed

set_seed(1)  # 设置随机种子


class MLP(nn.Module):
    def __init__(self, unit_nums, layer_nums=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(unit_nums, unit_nums, bias=False) for i in range(layer_nums)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(unit_nums) for i in range(layer_nums)])
        self.unit_nums = unit_nums

    def forward(self, x):
        for (i, linear), bn in zip(enumerate(self.linears), self.bn):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.mean()) or torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break

            print("layers:{},\t mean: {},\t std: {}".format(i, x.mean().item(), x.std().item()))

        return x

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # method 1
                # nn.init.normal_(m.weight.data, std=1)    # normal: mean=0, std=1

                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)


unit_nums = 256
layer_nums = 100
batch_size = 16

net = MLP(unit_nums, layer_nums)
# net.initialize()

inputs = torch.randn((batch_size, unit_nums))  # normal: mean=0, std=1

output = net(inputs)
print(output)

# -*- coding: utf-8 -*-
"""
数据迁移至cuda的方法
"""
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========================== tensor to cuda
flag = 0
# flag = 1
if flag:
    tensor_cpu = torch.ones((3, 3))
    print(
        "tensor_cpu:\ndevice: {} is_cuda: {} id: {}".format(
            tensor_cpu.device, tensor_cpu.is_cuda, id(tensor_cpu)
        )
    )

    tensor_gpu = tensor_cpu.to(device)
    print(
        "tensor_gpu:\ndevice: {} is_cuda: {} id: {}".format(
            tensor_gpu.device, tensor_gpu.is_cuda, id(tensor_gpu)
        )
    )

# ========================== module to cuda
flag = 0
# flag = 1
if flag:
    net = nn.Sequential(nn.Linear(3, 3))
    print("\nmodel_id:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

    net.to(device)
    print("\nmodel_id:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

# ========================== forward in cuda
flag = 0
# flag = 1
if flag:
    net = nn.Sequential(nn.Linear(3, 3)).to(device)
    tensor_gpu = torch.ones((3, 3)).to(device)
    output = net(tensor_gpu)
    print("output is_cuda: {}".format(output.is_cuda))

# ========================== 查看当前gpu 序号，尝试修改可见gpu，以及主gpu
flag = 0
# flag = 1
if flag:
    current_device = torch.cuda.current_device()
    print("current_device: ", current_device)

    torch.cuda.set_device(0)
    current_device = torch.cuda.current_device()
    print("current_device: ", current_device)

    #
    cap = torch.cuda.get_device_capability(device=None)
    print(cap)
    #
    name = torch.cuda.get_device_name()
    print(name)

    is_available = torch.cuda.is_available()
    print(is_available)

    # ===================== seed
    seed = 2
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    current_seed = torch.cuda.initial_seed()
    print(current_seed)

    s = torch.cuda.seed()
    s_all = torch.cuda.seed_all()

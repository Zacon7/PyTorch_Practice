# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from model.lenet import LeNet, LeNetSequential, LeNetSequentialOrderDict
from tools.my_dataset import RMBDataset
from enviroments import rmb_split_dir, rmb_test_dir
from tensorboardX import SummaryWriter


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()  # 设置随机种子
rmb_label = {"1": 0, "100": 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01
log_interval = 10
val_interval = 1

# ============================ step 1/5 数据 ============================
# 设置路径参数
train_dir = os.path.join(rmb_split_dir, "train")
valid_dir = os.path.join(rmb_split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 设置训练集的数据增强和转化
train_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.Grayscale(num_output_channels=3),  # 添加灰度变换，减少颜色带来的偏差，可以泛化到第五套人民币
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

# 设置验证集的数据增强和转化，不需要 RandomCrop
valid_transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),  # 添加灰度变换，减少颜色带来的偏差，可以泛化到第五套人民币
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ]
)

# 构建RMBDataset实例
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoader
# 其中训练集设置 shuffle=True，表示每个 Epoch 都打乱样本
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

net = LeNet(classes=2)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()  # 选择损失函数

# ============================ step 4/5 设置多学习率 ============================
LR_LIST = [0.01, 0.0125, 0.02]
train_loss_rec = [[] for _ in range(len(LR_LIST))]
valid_loss_rec = [[] for _ in range(len(LR_LIST))]

# ============================ step 5/5 训练 ============================
for i_lr, lr in enumerate(LR_LIST):
    print("-------------------- LR = {} ---------------------".format(lr))
    net.initialize_weights()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # 选择优化器
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

    train_curve = list()
    valid_curve = list()

    iterations = 0

    # 构建 SummaryWriter，用于 tensorvision 可视化数据
    # writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

    for epoch in range(MAX_EPOCH):
        loss_mean = 0.0
        correct = 0.0
        total = 0.0

        net.train()
        # 遍历 train_loader 取数据
        for i, data in enumerate(train_loader):
            iterations += 1
            # forward
            inputs, labels = data
            outputs = net(inputs)

            # backward
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print(
                    "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch,
                        MAX_EPOCH,
                        i + 1,
                        len(train_loader),
                        loss_mean,
                        correct / total,
                    )
                )
                loss_mean = 0.0

            # 记录数据，保存于event file
            # writer.add_scalars("Loss", {"Train": loss.item()}, iterations)
            # writer.add_scalars("Accuracy", {"Train": correct / total}, iterations)
        # 每个epoch，记录梯度，权值
        # for name, param in net.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)

        # scheduler.step()  # 每个 epoch 更新学习率
        # 每个 epoch 计算验证集得准确率和loss
        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.0
            total_val = 0.0
            loss_val = 0.0
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val / valid_loader.__len__())
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch,
                        MAX_EPOCH,
                        j + 1,
                        len(valid_loader),
                        loss_val,
                        correct_val / total_val,
                    )
                )

                # # 记录数据，保存于event file
                # writer.add_scalars("Loss", {"Valid": np.mean(valid_curve)}, iterations)
                # writer.add_scalars("Accuracy", {"Valid": correct / total}, iterations)

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    # 由于valid中记录的是epoch loss，需要对记录点进行转换到iterations
    valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval
    valid_y = valid_curve

    # plt.plot(train_x, train_y, label='Train')
    # plt.plot(valid_x, valid_y, label='Valid')
    #
    # plt.legend(loc='upper right')
    # plt.ylabel('loss value')
    # plt.xlabel('Iteration')
    # plt.show()

    train_loss_rec[i_lr] = train_curve
    valid_loss_rec[i_lr] = valid_curve

for i, loss_r in enumerate(train_loss_rec):
    ax1 = plt.subplot(121, title="Train Loss using different LR")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train Loss")
    ax1.plot(range(len(loss_r)), loss_r, label="LR = {}".format(LR_LIST[i]))
    ax1.legend()
    ax1.grid()
for i, loss_r in enumerate(valid_loss_rec):
    ax2 = plt.subplot(122, title="Valid Loss using different LR")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Valid Loss")
    ax2.plot(range(len(loss_r)), loss_r, label="LR = {}".format(LR_LIST[i]))
    ax2.legend()
    ax2.grid()
plt.show()
# # ============================ inference ============================
# test_data = RMBDataset(data_dir=rmb_test_dir, transform=valid_transform)
# valid_loader = DataLoader(dataset=test_data, batch_size=1)
# for i, data in enumerate(valid_loader):
#     # forward
#     inputs, labels = data
#     outputs = net(inputs)
#     _, predicted = torch.max(outputs.data, 1)
#
#     rmb = 1 if predicted.numpy()[0] == 0 else 100
#     print("模型获得{}元".format(rmb))

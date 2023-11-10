# -*- coding: utf-8 -*-
"""
模型finetune方法
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from lesson2.rmb_classification.tools.my_dataset import AntsDataset
from common_tools import set_seed
import torchvision.models as models
import enviroments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

set_seed(1)  # 设置随机种子
label_name = {"ants": 0, "bees": 1}

# 参数设置
MAX_EPOCH = 25
BATCH_SIZE = 16
LR = 0.001
log_interval = 10
val_interval = 1
classes = 2
start_epoch = -1
lr_decay_step = 7

# ============================ step 1/5 数据 ============================
data_dir = enviroments.hymenoptera_data_dir
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.64, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = AntsDataset(data_dir=train_dir, transform=train_transform)
valid_data = AntsDataset(data_dir=valid_dir, transform=valid_transform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

# 1/3 构建模型
resnet18_ft = models.resnet18()

# 2/3 加载预训练参数
# flag = 0
flag = 1
if flag:
    path_pretrained_model = enviroments.resnet18_path
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)

# finetune 法1 : 冻结卷积层参数
flag_m1 = 0
# flag_m1 = 1
if flag_m1:
    for param in resnet18_ft.parameters():
        param.requires_grad = False
    # print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))

# 3/3 替换fc层（原resnet的fc输出维度为1000，这里需要改为2）
# 首先拿到 fc 层的输入个数
num_fc_in = resnet18_ft.fc.in_features
# 然后构造新的 fc 层替换原来的 fc 层
resnet18_ft.fc = nn.Linear(num_fc_in, classes)
# 将模型转移到 gpu 上训练
resnet18_ft.to(device)

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()  # 选择损失函数

# ============================ step 4/5 优化器 ============================
# finetune 法2 : 卷积层参数设置小学习率
# flag_m2 = 0
flag_m2 = 1
if flag_m2:
    # 首先获取全连接层参数的地址
    fc_params_id = list(map(id, resnet18_ft.fc.parameters()))  # 返回的是parameters的 内存地址
    # 然后使用 filter 过滤不属于全连接层的参数，也就是保留卷积层的参数
    feature_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
    # 设置优化器的分组学习率，传入一个 list，包含 2 个字典元素。对应 2 个参数组
    optimizer = optim.SGD([{'params': feature_params, 'lr': LR * 0.1},
                           {'params': resnet18_ft.fc.parameters()}],
                          lr=LR, momentum=0.9)

else:
    optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)  # 选择优化器

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)  # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(start_epoch + 1, MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    resnet18_ft.train()
    for i, batch_data in enumerate(train_loader):

        # forward
        inputs, labels = batch_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18_ft(inputs)

        # backward
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()
        optimizer.zero_grad()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().cpu().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval  # 每 10 个batch打印一次 loss 信息
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}]\tLoss mean: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
            loss_mean = 0.

            # if flag_m1:
            # print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch + 1) % val_interval == 0:
        # 每个 epoch 都用验证集测试一下
        loss_val = 0.
        correct_val = 0.
        total_val = 0.

        resnet18_ft.eval()
        with torch.no_grad():  # 不用存储反向传播时需要的cache信息，节省显存
            for j, batch_data in enumerate(valid_loader):
                inputs, labels = batch_data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = resnet18_ft(inputs)
                loss = criterion(outputs, labels)

                loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

            loss_val_mean = loss_val / len(valid_loader)
            valid_curve.append(loss_val_mean)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}]\tLoss mean: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))
        resnet18_ft.train()

train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1,
                    len(valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epoch loss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

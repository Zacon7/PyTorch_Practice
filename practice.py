import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# 导入数据集的包
import torchvision.datasets
# 导入dataloader的包
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(10)


class FaceData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir

        self.img_path = os.path.join(self.root_dir, self.label_dir)
        self.img_list = os.listdir(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_path, img_name)
        img = Image.open(img_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_list)


def load_data():
    root_path = "D:/Documents/CodeProjects/Python/img-mark-tool-master/datasets/dataset"
    img_list = os.listdir(root_path)
    label_path = "D:/Documents/CodeProjects/Python/img-mark-tool-master/datasets/labels"
    label_dir = "tone"
    for img_name in img_list:
        if img_name.split('.')[1] == 'jpg':
            img_name = img_name.split('.')[0]
            with open(os.path.join(label_path, label_dir, img_name + ".txt"), 'w') as f:
                f.write(label_dir)


def test_torchvision():
    # 创建测试数据集
    test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False,
                                                transform=torchvision.transforms.ToTensor())
    # 创建一个dataloader,设置批大小为64，每一个epoch重新洗牌，不进行多进程读取机制，不舍弃不能被整除的批次
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

    writer = SummaryWriter("log")

    # loader中对象
    step = 0
    for data in test_dataloader:
        imgs, targets = data
        writer.add_images("loader", imgs, step)
        step += 1

    writer.close()

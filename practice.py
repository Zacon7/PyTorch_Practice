import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

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


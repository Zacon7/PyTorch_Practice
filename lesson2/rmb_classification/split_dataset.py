# -*- coding: utf-8 -*-

import os
import random
import shutil
from enviroments import project_dir


# 创建文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == "__main__":
    random.seed(1)

    dataset_dir = os.path.join(project_dir, "data", "RMB_data")  # 'data\\RMB_data'
    split_dir = os.path.join(project_dir, "data", "rmb_split")  # 'data\\rmb_split'
    train_dir = os.path.join(split_dir, "train")  # 'data\\rmb_split\\train'
    valid_dir = os.path.join(split_dir, "valid")  # 'data\\rmb_split\\valid'
    test_dir = os.path.join(split_dir, "test")  # 'data\\rmb_split\\test'

    # 训练集 80%
    train_pct = 0.8
    # 验证集 10%
    valid_pct = 0.1
    # 测试集 10%
    test_pct = 0.1

    dirs = os.listdir(dataset_dir)
    # dirs: ['1', '100']
    for sub_dir in dirs:
        # 文件列表
        imgs = os.listdir(os.path.join(dataset_dir, sub_dir))
        # 取出 jpg 结尾的文件
        imgs = list(filter(lambda x: x.endswith(".jpg"), imgs))
        random.shuffle(imgs)
        # 计算图片数量
        img_count = len(imgs)
        # 计算训练集索引的结束位置
        train_point = int(img_count * train_pct)
        # 计算验证集索引的结束位置
        valid_point = int(img_count * (train_pct + valid_pct))
        # 把数据划分到训练集、验证集、测试集的文件夹
        for i in range(img_count):
            if i < train_point:
                split_dir = os.path.join(train_dir, sub_dir)
            elif i < valid_point:
                split_dir = os.path.join(valid_dir, sub_dir)
            else:
                split_dir = os.path.join(test_dir, sub_dir)
            # 创建文件夹
            makedir(split_dir)
            # 构造目标文件名
            target_path = os.path.join(split_dir, imgs[i])
            # 构造源文件名
            src_path = os.path.join(dataset_dir, sub_dir, imgs[i])
            # 复制
            shutil.copy(src_path, target_path)

        print(
            "Class:{}, train:{}, valid:{}, test:{}".format(
                sub_dir, train_point, valid_point - train_point, img_count - valid_point
            )
        )

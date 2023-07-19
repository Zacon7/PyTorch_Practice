# -*- coding: utf-8 -*-

import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from common_tools import transform_invert, set_seed

set_seed(3)  # 设置随机种子

# ================================= load img ==================================
path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "lena.png")
img = Image.open(path_img).convert('RGB')  # 0~255

# convert to tensor
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
# 添加 batch 维度
img_tensor.unsqueeze_(dim=0)  # C*H*W to B*C*H*W

# ================================= create convolution layer ==================================
# input:(batch, in_c, h_in, w_in)  weights:(out_c, in_c, f, f)  output:(ba  tch, out_c, h_out, w_out)
# flag = True
flag = False
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)
    # 初始化卷积层权值
    # nn.init.xavier_normal_(conv_layer.weight.data)
    nn.init.xavier_uniform_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================================ transposed ================================
flag = True
# flag = False
if flag:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)
    # 初始化网络层的权值
    nn.init.xavier_uniform_(conv_layer.weight.data)

    # calculation
    img_conv = conv_layer(img_tensor)

# ================================= visualization ==================================
print("卷积前尺寸: {}\n卷积后尺寸: {}".format(img_tensor.shape, img_conv.shape))
img_raw = transform_invert(img_tensor[0, ...], img_transform)
img_conv = transform_invert(img_conv[0, ...], img_transform)
plt.subplot(121).imshow(img_raw)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.show()

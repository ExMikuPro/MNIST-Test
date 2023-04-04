#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : SFNCO-Studio
# @Time     : 2023/4/3 14:57
# @File     : model.py
# @Project  : Deep in Conlda
# @Uri      : https://sfnco.com.cn/


from main import Net

import torch
from mnist import *
import glob
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torchvision
from skimage import io, transform

if __name__ == '__main__':
    device = torch.device("mps")
    model = torch.load('./model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    print("模型加载完成")

    # 循环读取文件夹内的jpg图片并输出结果
    for jpgFile in glob.glob(r'./*.png'):
        print(jpgFile)
        img = cv2.imread(jpgFile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
        img = torch.from_numpy(img)
        img = img.to(device)
        output = model(Variable(img))
        prob = F.softmax(output, dim=1)
        prob = Variable(prob)
        prob = prob.cpu().numpy()
        print(prob)
        pred = np.argmax(prob)
        print(pred.item())
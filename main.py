#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : SFNCO-Studio
# @Time     : 2023/4/3 14:57
# @File     : main.py
# @Project  : Deep in Conlda
# @Uri      : https://sfnco.com.cn/

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from tqdm import tqdm

path = "./data/"

device = torch.device("mps")  # macos GPU加速

transForm = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

trainData = torchvision.datasets.MNIST(path, train=True, transform=transForm, download=True)

testData = torchvision.datasets.MNIST(path, train=False, transform=transForm)

BATCH_SIZE = 256

tranDataLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = Data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output


def main():
    net = Net()
    print(net.to(device))
    lossA = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    EPOCHS = 100

    history = {'Test Loss': [], 'Test Accuracy': []}

    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(tranDataLoader, unit='step')

        net.train(True)

        for step, (trainImg, labels) in enumerate(processBar):
            trainImg = trainImg.to(device)
            labels = labels.to(device)

            net.zero_grad()

            outputs = net(trainImg)

            loss = lossA(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]
            loss.backward()
            optimizer.step()

            processBar.set_description("[%d/%d] Loss:%.4f, Acc:%.4f" % (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                for testImgs, labels in testDataLoader:
                    testImgs = testImgs.to(device)
                    labels = labels.to(device)
                    outputs = net(testImgs)
                    loss = lossA(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)

                    totalLoss += loss
                    correct += torch.sum(predictions == labels)
                testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                testLoss = totalLoss / len(testDataLoader)
                history['Test Loss'].append(testLoss.item())
                history['Test Accuracy'].append(testAccuracy.item())
                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                            testAccuracy.item()))
                processBar.close()

    torch.save(net, './model.pth')


if __name__ == '__main__':
    main()

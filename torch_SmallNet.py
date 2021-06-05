#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Small Networks for Test
# Working Directory : D:\Work_2021\python_test_2021\pytorch_test_2021\
# Base URL     :
#
# 2021 05 27 by ***********
###########################################################################
_description = '''\
====================================================
torch_SmallNet.py : Based on torch module
                    Written by *********** @ 2021-03-10
====================================================
Example : python torch_SmallNet.py
'''

import torch

#-------------------------------------------------------------
# Description of CNN
# Input  : 1 channel 28x28 (28x28x1)
# Layer1 : conv2d (28x28x32) to pooling2d(14x14x32)
# Layer2 : conv2d (14x14x64) to pooling2d(7x7x64)
# FC     : plat(7x7x64) to 10
#-------------------------------------------------------------
class CNN(torch.nn.Module):
    def __init__(self, inputCH, outCH):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(inputCH, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = torch.nn.Linear(7 * 7 * 64, outCH, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#-------------------------------------------------------------
# Description of LeNet
# Reference :
# Input  : 1 channel 32x32x3
# Layer1 : conv2d (3x32x32x6) to pooling2d(16x16x6)
# Layer2 : conv2d (14x14x64) to pooling2d(7x7x64)
# FC     : plat(7x7x64) to 10
#-------------------------------------------------------------
class LeNet(torch.nn.Module):
    def __init__(self, inputCH, outCH):
        super(LeNet, self).__init__()
        self.LayerFeature = torch.nn.Sequential(
            torch.nn.Conv2d     (in_channels=inputCH, out_channels=6, kernel_size=(5, 5)),
            torch.nn.ReLU       (inplace=True),
            torch.nn.MaxPool2d  (kernel_size=(2,2), stride=2),
            torch.nn.Conv2d     (in_channels=6, out_channels=16, kernel_size=(5, 5)),
            torch.nn.ReLU       (inplace=True),
            torch.nn.MaxPool2d  (kernel_size=(2, 2), stride=2)
        )

        self.LayerClassify = torch.nn.Sequential(
            torch.nn.Linear(in_features= 5 * 5 * 16, out_features= 120),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=outCH),
            torch.nn.ReLU(),
            torch.nn.LogSoftmax(dim=-1)
        )

        # Network Initializaztion
        for m in self.LayerClassify:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        out = self.LayerFeature(x)
        out = out.view(out.size(0), -1)
        out = self.LayerClassify(out)
        return out


#-------------------------------------------------------------
# class for Service Function
#-------------------------------------------------------------



#-------------------------------------------------------------
# Test Main
#-------------------------------------------------------------
if __name__ == "__main__":
    print(" It is just test ")
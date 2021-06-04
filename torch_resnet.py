#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Definition of ResNet
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     : https://github.com/dnddnjs/pytorch-cifar10/blob/enas/resnet/model.py
#                https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
#                https://dnddnjs.github.io/cifar10/2018/10/09/resnet/
# ResNet
# 2021 03 25 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_resnet.py : Based on torch module
                    Written by Jinwuk @ 2021-03-25
====================================================
Example : This python file is not executable
'''
#=============================================================
# Definitions
#=============================================================
import torch.nn as nn
import torch.nn.functional as F

# code from https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityPadding, self).__init__()
        self.pooling        = nn.MaxPool2d(1, stride=stride)
        self.add_channels   = out_channels - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=False):
        super(ResidualBlock, self).__init__()
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)

        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.down_sample = IdentityPadding(in_channels, out_channels, stride) if down_sample else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = x if self.down_sample is None else self.down_sample(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=10, inputCH=3):
        super(ResNet, self).__init__()
        # First Input Image processing
        self.conv1  = nn.Conv2d(in_channels=inputCH, out_channels=16,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.relu   = nn.ReLU(inplace=True)

        # feature map size = 32x32x16 if stride=1, then Input channel and out channel is same
        self.layers_2n  = self.get_layers(block, 16, 16, stride=1, num_layers=num_layers[0])
        # feature map size = 16x16x32 because stride=2, 16x16 and Filters are 16x2=32
        self.layers_4n  = self.get_layers(block, 16, 32, stride=2, num_layers=num_layers[1])
        # feature map size = 8x8x64 because stride=2, 8x8 and Filters are 32x2=64
        self.layers_6n  = self.get_layers(block, 32, 64, stride=2, num_layers=num_layers[2])

        # output layers
        self.avg_pool   = nn.AvgPool2d(8, stride=1)
        self.fc_out     = nn.Linear(64, num_classes)

        # Weight Initilization
        self.total_layers   = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                self.total_layers += 1
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                self.total_layers += 1
            else:
                pass

    def get_layers(self, block, in_channels, out_channels, stride, num_layers):
        down_sample = True if stride == 2 else False
        layers_list = nn.ModuleList([block(in_channels, out_channels, stride, down_sample)])
        for _ in range(num_layers - 1):
            layers_list.append(block(out_channels, out_channels))

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers_2n(x)
        x = self.layers_4n(x)
        x = self.layers_6n(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

#-------------------------------------------------------------
# Service Function for Resnet Test
#-------------------------------------------------------------
def check_numlayers(num_layers):
    # Set the number of Layers
    L_num_layers = []
    if isinstance(num_layers, int):
        L_num_layers = [num_layers, num_layers, num_layers]
    elif isinstance(num_layers, list) and len(L_num_layers) == 3:
        L_num_layers = num_layers
    else:
        print("Error : Type of number of Layers is not correct !!!")
        exit()
    return L_num_layers


def resnet(num_layers=5):
    #Check Number of Layers
    L_num_layers = check_numlayers(num_layers)

    # total number of layers if 6n + 2. if n is 5 then the depth of network is 32.
    block = ResidualBlock
    model = ResNet(num_layers=L_num_layers, block=block)
    return model
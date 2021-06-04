#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Deep Neural Network Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     : https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
# 2020 07 02 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_nn01.py
                    Written by Jinwuk @ 2020-09-03
====================================================
Example : python torch_nn01.py 
'''
# import the necessary packages
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from torch_service import torch_service, MNISTDataSet

# =================================================================
# Definition of Network
# 여기에 있는 Definition은 데이터 1개에 대하여 수행하는 방식이다.
# =================================================================
class LeNet(torch.nn.Module):
    _out_nodes  = 10
    _target     = torch.zeros(_out_nodes, dtype=float)

    def __init__(self):
        super(LeNet, self).__init__()
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d     (in_channels=1, out_channels=20, kernel_size=(5, 5)),
            torch.nn.MaxPool2d  (kernel_size=(2,2), stride=2),
        )

        self.Layer2 = torch.nn.Sequential(
            torch.nn.Conv2d     (in_channels=20, out_channels=50, kernel_size=(5, 5)),
            torch.nn.MaxPool2d  (kernel_size=(2,2), stride=2),
        )

        self.LayerFC= torch.nn.Sequential(
            torch.nn.Linear(in_features= 4 * 4 * 50, out_features= 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, self._out_nodes),
            torch.nn.ReLU()
            #torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, _img):
        _state_1  = self.Layer1(_img)
        _state_2  = self.Layer2(_state_1)
        # Flat the _state for Full Connected Layer
        _state_v  = _state_2.view(_state_2.size(0), -1)     # make it the same shape as output
        _out      = self.LayerFC(_state_v)

        return _out

    # _labeldata[_idx] 를 입력으로 받아 이를 output node dimension과 동일한 Traget data로 만드는 함수
    def eval_target(self, _labeldata):
        _tgidx = int(_labeldata)
        self._target[_tgidx] = 1.0
        return self._target

    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# =================================================================
# Initilization
# =================================================================
# -----------------------------------------------------------------
# Set Global parameters
# -----------------------------------------------------------------
_batch_size =64
_idx        = 0

# -----------------------------------------------------------------
# Set Class
# -----------------------------------------------------------------
ts          = torch_service(_description)

# -----------------------------------------------------------------
# Read Data
# -----------------------------------------------------------------
print("Read LMDB file for MNIST Data")
_dataset    = ts._readLMDB(ts.args.LMDB_dir)
(_dataimg, _label) = _dataset[ts.args.dataid]
print("label : ", _label)
print("image : ", np.shape(_dataimg))

print("Load MNIST Data to DataLoader provided by torch")
train_data  = MNISTDataSet(ts._imgch, ts._imgheight, ts._imgwidth, _dataset, transform=transforms.ToTensor())
trainloader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)
_iter       = iter(trainloader)
_data       = _iter.__next__()
_imgdata    = _data['image']
_labeldata  = _data['label']

# -----------------------------------------------------------------
# Set Network
# -----------------------------------------------------------------
_net        = LeNet()
criterion   = torch.nn.MSELoss()

print("============================================")
print(_net)
print("============================================")
print(" Information ")
print("--------------------------------------------")
print("Label Data : \n", _labeldata)
print("Size : %3d ID : %3d  Class: %3d " %(len(_labeldata), _idx, int(_labeldata[_idx])))
print("batch_size : %3d " %_batch_size)
# =================================================================
# Test Code
# =================================================================
#for _idx in range(_batch_size):
# one data code
x = _imgdata[_idx]
y = _net.forward(x)

# loss_a = (y - t).pow(2).sum() 와 비교해 보면 해당 loss는 Dimension(여기서는 10)을 나누어 준것이다.
t       = _net.eval_target(_labeldata[_idx])
loss    = criterion(y, t)

print("[%d] " %_idx, "Inference : ", y.detach().numpy(), " Real : %d  Loss : %f " %(int(_labeldata[_idx]), loss))

# =================================================================
# Processing
# Data Dimension must be (batchsize, channel, height, width)
# =================================================================







# =================================================================
# Finish\
# =================================================================
print("Processinf Finished !!")


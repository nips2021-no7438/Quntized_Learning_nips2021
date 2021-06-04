#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     : https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
# 2020 07 02 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_mlp.py
                    Written by Jinwuk @ 2020-07-03
====================================================
Example : python opencv_object_tracking.py --tracker kcf
'''

#=============================================================
# Definitions
#=============================================================
import matplotlib.pyplot as plt
import os
import torch

# Private Data
_loss_trend = []
_file_name  = 'torch_data.tcdat'

#=============================================================
# Processing
#=============================================================
print(_description)
print("-------------------------------------------------------")
print(" 0. Data Setting")
print("-------------------------------------------------------")
print(" Working Path : ", os.getcwd())

# System Data
dtype   = torch.float
device  = torch.device("cpu")
# device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.
N, D_in, H, D_out = 64, 1000, 100, 10

print("-------------------------------------------------------")
print(" 1. Data Generation")
print("-------------------------------------------------------")
if os.path.isfile(_file_name):
    [x, y, w1, w2] = torch.load(_file_name)
    print("There exists Torch Data File : ", _file_name)
else:
    x  = torch.randn(N, D_in, device=device, dtype=dtype)
    y  = torch.randn(N, D_out, device=device, dtype=dtype)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    torch.save([x, y, w1, w2], _file_name)
    print("Save Torch Data File : ", _file_name)

print("-------------------------------------------------------")
print(" 2. Learning")
print("-------------------------------------------------------")
learning_rate = 1e-6
for t in range(500):
    # Forward
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99: print("[%4d] %f" %(t, loss.item()))

    #Record the loss
    _loss_trend.append(loss.item())
    # backward
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Set zero grad after update
        w1.grad.zero_()
        w2.grad.zero_()

#=============================================================
# Final Stage
#=============================================================
print("-------------------------------------------------------")
print(" 3. Finish the Program")
print("-------------------------------------------------------")
print("Number of EPOCH", len(_loss_trend))
plt.plot(_loss_trend[:20])
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.show()

print("Process Finished!!")
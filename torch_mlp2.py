#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     :
# 2020 07 02 by ***********
###########################################################################
_description = '''\
====================================================
torch_mlp2.py : Based on torch module
                    Written by *********** @ 2021-03-08
====================================================
Example : python torch_mlp2.py
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
_model_name = 'mlp2_model.pt'
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
# device = torch.device("cuda:0") # Please remove the comment if you want operate the code on GPU.
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

print("Input Size :", x.shape)
print("Output Size:", y.shape)
print("w1 Size    :", w1.shape)
print("w2 Size    :", w2.shape)

#-------------------------------------------------------------
# Torch Processing
#-------------------------------------------------------------
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

# Definition of Layer (For various test)
linear1 = torch.nn.Linear(D_in, H, bias=True)
linear2 = torch.nn.Linear(H, D_out, bias=True)
relu    = torch.nn.ReLU()

#Set Model and Initilization
model = torch.nn.Sequential(linear1, relu, linear2)
model.apply(init_weights)

# Set Loss
loss_fn = torch.nn.MSELoss(reduction='sum')

print("-------------------------------------------------------")
print(" 2. Learning")
print("-------------------------------------------------------")
# When you use nn Module, LR=1.0 is equal to 0.01. Therefore, set LR=10^-4 to be 10^-6.
learning_rate = 1e-4
for Epoch in range(100):
    # Forward
    y_pred  = model(x)
    loss    = loss_fn(y_pred, y)

    #Record the loss
    if Epoch % 10 == 9: print("[%4d] %f" %(Epoch, loss.item()))
    _loss_trend.append(loss.item())

    # backward :
    # 1. Set all gradient to be Zero
    # 2. General Backward
    model.zero_grad()
    loss.backward()

    # Weight Update
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

#=============================================================
# Final Stage
#=============================================================
torch.save(model, _model_name)

print("size of loss", len(_loss_trend))
plt.plot(_loss_trend[:20])
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.show()

print("Process Finished!!")
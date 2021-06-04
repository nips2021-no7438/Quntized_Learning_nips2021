#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     : https://wikidocs.net/61010
# Torch NN Module in class
# 2020 07 02 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_mlp3.py : Based on torch module
                    Written by Jinwuk @ 2021-03-10
====================================================
Example : python torch_mlp3.py
'''

#=============================================================
# Definitions
#=============================================================
import matplotlib.pyplot as plt
import os
import torch

class Torch_MLP(torch.nn.Module):
    def __init__(self):
        self.N     = 64
        self.D_in  = 1000
        self.H     = 100
        self.D_out = 10
        super(Torch_MLP, self).__init__()
        self.Layer = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H, self.D_out, bias=True)
        )

    def forward(self, x):
        out = self.Layer(x)
        return out

#=============================================================
# Test Processing
#=============================================================
if __name__ == "__main__":
    # Private Data
    _loss_trend = []
    _file_name  = 'torch_data.tcdat'
    _model_name = 'mlp3_model.pt'
    #=============================================================
    # Processing
    #=============================================================
    print(_description)
    print("-------------------------------------------------------")
    print(" 0. Data Setting")
    print("-------------------------------------------------------")
    print(" Working Path : ", os.getcwd())
    print("-------------------------------------------------------")
    print(" 1. Data Generation")
    print("-------------------------------------------------------")

    x, y = 0, 0
    if os.path.isfile(_file_name):
        [x, y, w1, w2] = torch.load(_file_name)
        print("There exists Torch Data File : ", _file_name)
    else:
        print("There is not Torch Data File : ", _file_name)

    print("Input Size :", x.shape)
    print("Output Size:", y.shape)

    #-------------------------------------------------------------
    # Torch Processing
    #-------------------------------------------------------------
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model = Torch_MLP()
    model.apply(init_weights)

    # Set Loss
    loss_fn = torch.nn.MSELoss(reduction='sum')

    print("-------------------------------------------------------")
    print(" 2. Learning")
    print("-------------------------------------------------------")
    # nn Module을 사용할 경우 LR=1.0 은 실제로는 0.01 이 된다. 고로 10^-4을 해야 원래대로 10^-6이 된다.
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
    torch.save(model.state_dict(), _model_name)

    print("size of loss", len(_loss_trend))
    plt.plot(_loss_trend[:20])
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.show()

    print("Process Finished!!")
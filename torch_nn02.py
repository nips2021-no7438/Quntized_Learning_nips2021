#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     :
# Simple CNN for MNIST test (Truth test)
# 2020 07 02 by ***********
###########################################################################
_description = '''\
====================================================
torch_nn02.py : Based on torch module
                    Written by *********** @ 2021-03-10
====================================================
Example : python torch_nn02.py 
'''

#=============================================================
# Definitions
#=============================================================
import matplotlib.pyplot as plt
import os
import pickle
import torch
import time
from Read_Data import MNIST_set
from Read_Data import CIFAR10_set
#-------------------------------------------------------------
# Description of CNN, LeNet, ResNet
# Reference : https://github.com/dnddnjs/pytorch-cifar10/blob/enas/resnet/model.py
# Input  : 3 channel 32x32x3
#-------------------------------------------------------------
from torch_SmallNet import CNN
from torch_SmallNet import LeNet
from torch_resnet import ResNet as resnet_base
from torch_resnet import ResidualBlock
import torch_resnet as resnet_service

def ResNet(inputCH, outCH, num_layers=5):
    Lnum_layers = resnet_service.check_numlayers(num_layers)
    block = ResidualBlock
    model = resnet_base(num_layers=Lnum_layers, block=block, num_classes=outCH, inputCH=inputCH)
    return model

#=============================================================
# Function for Test Processing
#=============================================================
g_msg = []
# --------------------------------------------------------
# Service Function
# --------------------------------------------------------
def _sprint(msg):
    g_msg.append(msg)
    print(msg)

def _write_operation(opPATH):
    with open(opPATH, 'wt') as f:
        for _msg in g_msg:
            f.write(_msg + "\n")
    print("Operation Result File : %s " %opPATH)

def Check_modelName(args):
    l_algorithm = ['SGD', 'Adam', 'AdamW', 'QSGD', 'QtAdamW']
    b_correct = False
    for _name in l_algorithm:
        if args.model_name == _name:
            b_correct = True
            break
        else: pass

    if b_correct:
        print("Correct Model Name [{}]".format(args.model_name))
    else:
        print("Unexpected Model Name!! [{}]".format(args.model_name))
        print("Please Check the Model Name !!!")
        exit()

def Set_Data_Processing(args, device):
    if args.data_set == 'CIFAR10':
        Dset    = CIFAR10_set(batch_size=args.batch_size, bdownload=True)
        if args.net_name == 'ResNet':
            model  = ResNet(inputCH=Dset.inputCH, outCH=Dset.outputCH, num_layers=args.num_resnet_layers).to(device)
        else:
            model  = LeNet(inputCH=Dset.inputCH, outCH=Dset.outputCH).to(device)
    elif args.data_set == 'MNIST':
        Dset    = MNIST_set(batch_size=args.batch_size, bdownload=True)
        model   = CNN(inputCH=Dset.inputCH, outCH=Dset.outputCH).to(device)
    else:
        Dset, model = None, None
        print("Data set is not assigned !! It is Error!!!")
        exit()
    return Dset, model

# --------------------------------------------------------
# Parsing the Argument
# --------------------------------------------------------
import argparse
import textwrap

def _ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='torch_nn02.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-g', '--device', help="Using [0]CPU or [1]GPU",
                        type=int, default=0)
    parser.add_argument('-l', '--learning_rate', help="learning_rate",
                        type=float, default=0.001)
    parser.add_argument('-e', '--training_epochs', help="training_epochs",
                        type=int, default=15)
    parser.add_argument('-b', '--batch_size', help="batch_size",
                        type=int, default=100)
    parser.add_argument('-f', '--model_file_name', help="model file name",
                        type=str, default="torch_nn02_CNN.pt")
    parser.add_argument('-m', '--model_name', help="model name 'SGD', 'Adam', 'AdamW', 'QSGD'",
                        type=str, default="Adam")
    parser.add_argument('-n', '--net_name', help="Network name 'CNN', 'LeNet', 'ResNet'",
                        type=str, default="LeNet")
    parser.add_argument('-d', '--data_set', help="data set 'MNIST', 'CIFAR10'",
                        type=str, default="MNIST")
    parser.add_argument('-a', '--autoproc', help="Automatic Process without any plotting",
                        type=int, default=0)
    parser.add_argument('-pi', '--proc_index_name', help="Process Index Name. It is generated automatically",
                        type=str, default='')
    parser.add_argument('-rl', '--num_resnet_layers', help="The number of layers in a block to ResNet",
                        type=int, default=5)
    parser.add_argument('-qp', '--QParam', help="Quantization Parameter",
                        type=int, default=0)

    args = parser.parse_args()
    args.batch_size = 128 if args.data_set == 'CIFAR10' else 100
    args.autoproc   = True if args.autoproc == 1 else False
    args.net_name   = 'CNN' if args.data_set == 'MNIST' else args.net_name
    args.proc_index_name = args.net_name + args.model_name + str(args.training_epochs)

    Check_modelName(args)
    _sprint(_intro_msg)
    return args

# --------------------------------------------------------
# Processing Function
# --------------------------------------------------------
from torch_learning import learning_module

def _operation():
    # Set Test Processing
    _args   = _ArgumentParse(_description)
    _device = 'cuda' if _args.device == 1 and torch.cuda.is_available() else 'cpu'

    # Set Data Processing
    Dset, model = Set_Data_Processing(_args, _device)

    # Set Operation
    _args.model_file_name   = 'torch_nn02' + _args.proc_index_name + '.pt'
    _error_trend_file       = 'error_' + _args.proc_index_name + '.pickle'

    LoadingData     = Dset.data_loader(bTrain=True, bsuffle=False)
    _total_batch    = len(LoadingData)
    criterion       = torch.nn.CrossEntropyLoss()
    optimizer       = learning_module(model=model, args=_args)
    c_opt           = optimizer.optimizer

    _sprint("Data Set              : %s" %(_args.data_set))
    _sprint('Total number of Batch : {}'.format(_total_batch))
    _sprint("Batch SIze            : %d" %(Dset.batchsize))
    _sprint("Dimension of Data     : {}".format(Dset.datashape))
    _sprint("Hardware Platform     : %s" %(_device))
    _sprint("Model File Name       : %s" %(_args.model_file_name))
    _sprint("Error Trend File Name : %s" %(_error_trend_file))
    _sprint("Learning algorithm    : %s" %_args.model_name)
    _sprint("Learning rate         : {}".format(_args.learning_rate))
    if _args.model_name == 'QSGD' or _args.model_name == 'QtAdamW':
        _sprint("Initial Quantize Index: {}".format(_args.QParam))
        _sprint("Initial QP            : {}".format(c_opt.Q_proc.c_qtz.get_QP()))
    if _args.net_name == 'ResNet':
        _sprint("ResNet num. of Layers : %d" % (model.total_layers))
    _sprint("\n")

    # --------------------------------------------------------
    # Learning : X input, Y Label or Target using LoadingData based on Torch's data_loader
    # --------------------------------------------------------
    _loss_trend = []
    _start_time = time.time()
    for epoch in range(_args.training_epochs):
        _avg_cost, _k = 0, 0
        for X, Y in LoadingData:
            # Data loading on CPU or GPU
            X, Y = X.to(_device), Y.to(_device)

            # The part without any connection to Learning is operated as the state of Gradient=0
            optimizer.zero_grad()
            _prediction = model.forward(X)
            _cost = criterion(_prediction, Y)
            _cost.backward()

            # Learning : check at the following function
            optimizer.learning(epoch)

            # Update Index to batch
            _avg_cost   += _cost/_total_batch
            _k          += 1

        _loss_trend.append(_avg_cost)
        if _args.model_name == 'QSGD' or _args.model_name == 'QtAdamW':
            _sprint("[Epoch : %4d] cost = %f   QP Index: %d  Inf_QP: %d " \
                    %(epoch, _avg_cost, c_opt.Q_proc.Get_QPIndex(), c_opt.Q_proc.Get_InfQPIndex()) )
        else:
            _sprint("[Epoch : %4d] cost = %f" %(epoch, _avg_cost))

    _Learning_time = time.time() - _start_time
    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------
    # without any learning so that we operate the test under torch.no_grad()
    with torch.no_grad():
        _total, _correct, _accuracy = Dset.Test_Function(model, _device)

        _sprint("-----------------------------------------------------------------")
        _sprint("Total samples : %d   Right Score : %d " %(_total, _correct))
        _sprint("Accuracy      : %f" %_accuracy)
        _sprint("-----------------------------------------------------------------")
        _sprint("Total Learning Time   : %.3f sec"  %(_Learning_time))
        _sprint("Average Learning Time : %.3f sec" %(_Learning_time/_args.training_epochs))
    # --------------------------------------------------------
    # Final Stage
    # --------------------------------------------------------
    torch.save(model.state_dict(), _args.model_file_name)
    with open(_error_trend_file, 'wb') as f:
        pickle.dump(_loss_trend, f)

    return _args
# =============================================================
# Test Processing
# =============================================================
if __name__ == "__main__":
    params = _operation()

    _opfilename = 'operation_' + params.proc_index_name + '.txt'
    _write_operation(_opfilename)

    _sprint("Process Finished!!")

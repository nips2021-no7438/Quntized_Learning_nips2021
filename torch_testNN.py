#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Multilayer Perceptron Test
# Working Directory : D:\Work_2020\[099]_git_test\python_test_2020\pytorch_test
# Base URL     : https://eda-ai-lab.tistory.com/404
# 2021 03 09 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_testNN.py : Based on torch module
                    Written by Jinwuk @ 2021-03-12
====================================================
Example : python torch_testNN.py
'''

#=============================================================
# Definitions
#=============================================================
import matplotlib.pyplot as plt
import os
import pickle
import torch
import argparse
import textwrap
import time
from torch_testmodel import torch_test

import torch_nn02

class NNmodels:
    def __init__(self, args, inputCH, outCH):
        self.model = 0
        if args.modelclass == "CNN":
            self.model      = torch_nn02.CNN(inputCH, outCH)
        elif args.modelclass == "LeNet":
            self.model      = torch_nn02.LeNet(inputCH, outCH)
        elif args.modelclass == "ResNet":
            self.model      = torch_nn02.ResNet(inputCH, outCH, num_layers=args.num_resnet_layers)
        else:
            print("NN module is not dipicted!!! It is error ")
            print("model class : ", args.modelclass)
            print("model file  : ", args.modelfile)
            exit()

# =================================================================
# Parsing the Argument
# =================================================================
def ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='test pytorch_inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-g', '--device', help="Using [0]CPU or [1]GPU",
                        type=int, default=0)
    parser.add_argument('-d', '--dataset', help="Name of Data SET 'MNIST', 'CIFAR10'",
                        type=str, default='MNIST')
    parser.add_argument('-m', '--modelfile', help="Name of Model file",
                        type=str, default='torch_nn02_CNN.pt')
    parser.add_argument('-n', '--modelclass', help="Name of Model class 'CNN', 'LeNet', 'ResNet'",
                        type=str, default='CNN')
    parser.add_argument('-e', '--error_trend_file', help="Error Trend File such as 'error_file.pickle'",
                        type=str, default='error_file.pickle')
    parser.add_argument('-ng', '--no_graph', help="Plot the erroe trend [1] of learning or not[0 : default]",
                        type=int, default=0)
    parser.add_argument('-p', '--plotting_points', help="plotting_points for Error Trend ",
                        type=int, default=20)
    parser.add_argument('-rl', '--num_resnet_layers', help="The number of layers in a block to ResNet",
                        type=int, default=5)

    args = parser.parse_args()
    args.no_graph = True if args.no_graph == 1 else False
    print(_intro_msg)
    return args

def TensorList_to_numpyList(pk_loss_trend):
    _loss_trend = []
    for _data in pk_loss_trend:
        _a = _data.to('cpu')
        _b = _a.detach()
        _c = _b.numpy()
        _loss_trend.append(_c)
    return _loss_trend

# =================================================================
# Operation
# =================================================================
def Operation(L_Param, bUseParam=False):
    # -----------------------------------------------------------------
    # Definition
    # -----------------------------------------------------------------
    _args    = ArgumentParse(_description)
    _device  = _args.device
    _sprint  = torch_nn02._sprint
    # -----------------------------------------------------------------
    # Data Setting
    # -----------------------------------------------------------------
    Dset, TrDset, TeDset = [], [], []
    if _args.dataset == 'MNIST':
        from Read_Data import MNIST_set
        Dset        = MNIST_set(batch_size=100, bdownload=True)
        TrDset      = Dset.data_loader(bTrain=True, bsuffle=False)
        TeDset      = Dset.mnist_test
    elif _args.dataset == 'CIFAR10':
        from Read_Data import CIFAR10_set
        Dset        = CIFAR10_set(batch_size=4, bdownload=True)
        TrDset      = Dset.data_loader(bTrain=True, bsuffle=False)
        TeDset      = Dset.data_loader(bTrain=False, bsuffle=False)
    else:
        print("Data set is not depicted. It is error!!!")
        exit()

    _total_batch = len(TrDset)

    # -----------------------------------------------------------------
    # NN Model Setting
    # -----------------------------------------------------------------
    NNmodule    = NNmodels(_args, inputCH=Dset.inputCH, outCH=Dset.outputCH)
    model       = NNmodule.model.to(_device)
    model.load_state_dict(torch.load(_args.modelfile))
    model.eval()
    modelname   = str(model.__class__.__name__)

    print("-----------------------------------------------------------------")
    print("Model : ", model)
    print("-----------------------------------------------------------------")
    print("Total number of batch : %d" %len(TrDset))
    print("Data Set              : %s" %(_args.dataset))
    print("Batch SIze            : %d" %(Dset.batchsize))
    print("Dimension of Data     : {}".format(Dset.datashape))
    print("Hardware Platform     : %s" %(_device))
    print("Model File Name       : %s" %(_args.modelfile))
    print("Error Trend File Name : %s" %(_args.error_trend_file))
    if modelname == 'ResNet':
        _sprint("ResNet num. of Layers : %d" % (model.total_layers))

    print("-----------------------------------------------------------------")
    _start_time = time.time()
    # -----------------------------------------------------------------
    # Check the performance Ref : https://wikidocs.net/63565
    # -----------------------------------------------------------------
    with torch.no_grad():
        _total, _correct, _accuracy = Dset.Test_Function(model, _device, bTrain=True)
        _test_time_4_training       = time.time() - _start_time
        _sprint("-----------------------------------------------------------------")
        _sprint("Test Mode     : Trainning" )
        _sprint("Total samples : %d   Right Score : %d " %(_total, _correct))
        _sprint("Accuracy      : %f" %_accuracy)
        _sprint("Total time    : %f" %_test_time_4_training)
        _sprint("Average time  : %f" %(_test_time_4_training/_total))

        _total, _correct, _accuracy = Dset.Test_Function(model, _device, bTrain=False)
        _test_time_4_testing        = time.time() - _start_time
        _sprint("-----------------------------------------------------------------")
        _sprint("Test Mode     : Testing" )
        _sprint("Total samples : %d   Right Score : %d " %(_total, _correct))
        _sprint("Accuracy      : %f" %_accuracy)
        _sprint("Total time    : %f" %_test_time_4_testing)
        _sprint("Average time  : %f" %(_test_time_4_testing/_total))
        _sprint("-----------------------------------------------------------------")

    # -----------------------------------------------------------------
    # Processing Finished
    # -----------------------------------------------------------------
    if os.path.isfile(_args.error_trend_file) and _args.no_graph is not True:
        with open(_args.error_trend_file, 'rb') as f:
            pk_loss_trend = pickle.load(f)
        _loss_trend = TensorList_to_numpyList(pk_loss_trend)
        _n_point = len(_loss_trend) if len(_loss_trend) < _args.plotting_points else _args.plotting_points

        plt.plot(_loss_trend[:_n_point])
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.show()
    else:
        if os.path.isfile(_args.error_trend_file) is not True:
            print("There is not any error_trend_file")
        elif _args.no_graph is True:
            print("No graph is set")
        else:
            print("Because unknown reason, the error graph is not plotted")

    # -----------------------------------------------------------------
    # Write out the result as file
    # -----------------------------------------------------------------
    if len(L_Param) > 0:
        _file_name = L_Param[0]
        torch_nn02._write_operation(_file_name)

    return _args
# =================================================================
# Test Processing
# =================================================================
if __name__ == "__main__":
    _operation_param = []
    _operation_param.append('operation_test.txt')

    Operation(_operation_param)

    print("Processing is finished")

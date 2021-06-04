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
torch_testmodel.py : Based on torch module
                    Written by Jinwuk @ 2021-03-08
====================================================
Example : python torch_testmodel.py
'''

#=============================================================
# Definitions
#=============================================================
import matplotlib.pyplot as plt
import os
import torch
import argparse
import textwrap

class torch_test:
    def __init__(self, _datafile, _modelfile):
        self._file_name  = _datafile
        self._model_name = _modelfile

    def load_rddata(self):
        _x, _y = 0, 0
        if os.path.isfile(self._file_name):
            [_x, _y, w1, w2] = torch.load(self._file_name)
            print("There exists Torch Data File : ", self._file_name)
        else:
            print("There is not data file : ", self._file_name)
            exit()

        return _x, _y

    # Based on Simple Model
    def model_init(self):
        self.model       = torch.load(self._model_name)
        self.model.eval()

# =================================================================
# Parsing the Argument
# =================================================================
def _ArgumentParse(_intro_msg):
    parser = argparse.ArgumentParser(
        prog='test pytorch_inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(_intro_msg))

    parser.add_argument('-g', '--usingGPU', help="CPU or GPU based processing [0] CPU (default) [>1] GPU",
                        type=int, default=0)
    parser.add_argument('-d', '--datafile', help="Name of Data and Label data file",
                        type=str, default='torch_data.tcdat')
    parser.add_argument('-m', '--modelfile', help="Name of Model file",
                        type=str, default='mlp2_model.pt')
    parser.add_argument('-n', '--modelclass', help="Name of Model class",
                        type=str, default='')

    args = parser.parse_args()
    args.usingGPU = False if args.usingGPU == 0 else True

    return args

# =================================================================
# Test Processing
# =================================================================
if __name__ == "__main__":
    _args   = _ArgumentParse(_description)
    test_nn = torch_test(_args.datafile, _args.modelfile)
    _datum, _labels = test_nn.load_rddata()

    if _args.modelfile == '':
        test_nn.model_init()
        _prediction     = test_nn.model(_datum)
    else:
        if _args.modelfile == "mlp3_model.pt":
            from torch_mlp3 import Torch_MLP
            model = Torch_MLP()
        else:
            model = []
            print("Model is not correctly depicted!!")
            exit()

        model.load_state_dict(torch.load(_args.modelfile))
        model.eval()
        _prediction = model.forward(_datum)

    print("--------------------------------------------------------")
    print("Dimensions")
    print("data       :", _datum.shape)
    print("label      :", _labels.shape)
    print("prediction :", _prediction.shape)
    print("--------------------------------------------------------")

    Accuracy = 0
    for _k in range(len(_datum)):
        _real = torch.argmax(_labels[_k])
        _infr = torch.argmax(_prediction[_k])
        Accuracy += 1 if _real == _infr else 0
        #print("%d : %d  %d  %d" %(_k, _real, _infr, Accuracy))

    _Result = 100.0 * (Accuracy/(1.0 * len(_datum)))
    print("Result     :", _Result)
# -----------------------------------------------------------------
# Processing Finished
# -----------------------------------------------------------------
    print("Processing is finished")



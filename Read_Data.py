#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Service Function
#
# 2020 07 02 by ***********
###########################################################################
_description = '''\
====================================================
Read_Data.py
                    Written by ******* @ 2021-03-10
====================================================
for Reading of data set for Torch testbench  
'''
#=============================================================
# Definitions
#=============================================================
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

from torch.utils.data import TensorDataset  # Tensor Data Set
#from torch.utils.data import DataLoader     # Data loader

# For fix codes the MNIST data loading
'''
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
'''

g_datadir     = 'Torch_Data/'

class MNIST_set:
    def __init__(self, batch_size, bdownload):
        self.batchsize   = batch_size
        self.datadir     = g_datadir
        self.mnist_train = dsets.MNIST(root=self.datadir,  # 다운로드 경로 지정
                         train      =True,                  # True를 지정하면 훈련 데이터로 다운로드
                         transform  =transforms.ToTensor(), # 텐서로 변환
                         download   =bdownload)

        self.mnist_test  = dsets.MNIST(root=self.datadir,  # 다운로드 경로 지정
                         train      =False,                 # False를 지정하면 테스트 데이터로 다운로드
                         transform  =transforms.ToTensor(), # 텐서로 변환
                         download   =bdownload)
        self.inputCH     = 1
        self.datashape   = self.mnist_test.data.shape[1:2]
        self.outputCH    = len(self.mnist_train.classes)


    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset    = self.mnist_train if bTrain else self.mnist_test
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                    batch_size =self.batchsize,
                    shuffle    =bsuffle,
                    drop_last  =bdrop_last)
        return loadingData

    def TestData_loader(self):
        X_test = self.mnist_test.data.view(len(self.mnist_test), 1, 28, 28).float()
        Y_test = self.mnist_test.targets
        return X_test, Y_test

    def Test_Function(self, model, _device, ClassChk=False):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            # For MNIST : Data loading on CPU or GPU
            _X, _Y = self.TestData_loader()
            _X, _Y = _X.to(_device), _Y.to(_device)

            _prediction = model(_X)
            _correct_chk= torch.argmax(_prediction, 1) == _Y
            _score      = _correct_chk.float().mean()
            _total      = len(_correct_chk)
            _correct    = _correct_chk.sum()
            _accuracy   = _score.item()

            return _total, _correct, _accuracy

class CIFAR10_set:
    # batch size = 4 추천
    def __init__(self, batch_size, bdownload):
        self.batchsize  = batch_size
        self.datadir    = g_datadir
        self.classes    = ('plane', 'car', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck')

        self.transform  = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset   = dsets.CIFAR10(root=self.datadir,
                                        train=True,
                                        transform=self.transform,
                                        download=bdownload)
        self.testset    = dsets.CIFAR10(root=self.datadir,
                                        train=False,
                                        transform=self.transform,
                                        download=bdownload)
        self.inputCH     = 3
        self.datashape   = self.trainset.data.shape
        self.outputCH    = len(self.trainset.classes)

    # Train의 경우 suffle은 True, Test의 경우 False 추천
    def data_loader(self, bTrain, bsuffle, bdrop_last=True):
        pdataset = self.trainset if bTrain else self.testset
        loadingData = torch.utils.data.DataLoader(dataset=pdataset,
                                                  batch_size=self.batchsize,
                                                  shuffle=bsuffle,
                                                  num_workers=2,
                                                  drop_last=bdrop_last)
        return loadingData

    def Test_Function(self, model, _device, ClassChk=False, bTrain=False ):
        if ClassChk:
            print("Check the Classification Accuracy per Class is not implemented yet")
            exit()
        else:
            _total, _correct = 0, 0
            LoadingData = self.data_loader(bTrain=bTrain, bsuffle=False)
            #for _data in LoadingData:
            for _X, _Y in LoadingData:
                #_X, _Y = _data
                _X, _Y = _X.to(_device), _Y.to(_device)
                _prediction = model(_X)
                _value, _predicted = torch.max(_prediction.data, 1)
                _total += _Y.size(0)
                _correct += (_predicted == _Y).sum().item()

            _accuracy = _correct / _total

            return _total, _correct, _accuracy

#=============================================================
# Test Processing
#=============================================================
if __name__ == "__main__":
    # =================================================================
    # Parsing the Argument
    # =================================================================
    import argparse
    import textwrap

    def _ArgumentParse(_intro_msg):
        parser = argparse.ArgumentParser(
            prog='test pytorch_inference',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(_intro_msg))

        parser.add_argument('-d', '--dataset',
                            help="Name of Data SET 'MNIST', 'CIFAR10'",
                            type=str, default='MNIST')
        parser.add_argument('-t', '--training',
                            help="[0] test [1] training",
                            type=int, default=1)
        args = parser.parse_args()
        args.training = True if args.training == 1 else False
        return args
    #=============================================================
    # Test Processing
    #=============================================================
    _args = _ArgumentParse(_description)

    if _args.dataset == 'MNIST':
        # For fix codes the MNIST data loading
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        Dset        = MNIST_set(batch_size=100, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    elif _args.dataset == 'CIFAR10':
        Dset        = CIFAR10_set(batch_size=4, bdownload=True)
        LoadingData = Dset.data_loader(bTrain=True, bsuffle=False)
    else:
        LoadingData = []
        print("Data set is not depicted. It is error!!!")
        exit()

    print("Total number of batch : ", len(LoadingData))

    K = 0
    for X, Y in LoadingData:
        if K == 0:
            print("Data <dim> : ", X.shape)
            print("Label<dim> : ", Y.shape)
        K+= 1

    #=============================================================
    # Final Stage
    #=============================================================

    print("Process Finished!!")

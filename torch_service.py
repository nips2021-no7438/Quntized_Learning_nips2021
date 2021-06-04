#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Service Function
#
# Base URL     : https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
# 2020 07 02 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_service.py
                    Written by Jinwuk @ 2020-07-03
====================================================
only for test of the torch_service.py 
'''

import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
import lmdb
import numpy as np

import os
import sys
import platform

import argparse
import textwrap

# =================================================================
# Input List of label and List of Image data composed with numpy array
# transforms.Normalize(mean=_mean, std=_std)
# =================================================================
class   MNISTDataSet:
    def __init__(self, _imgch, _imgheight, _imgwidth, _dataset, transform=None):
        self._transform = transform
        #self._normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self._normalize = MinMaxScaler()
        self._labellist = []
        self._imglist   = []

        print("Conversion MNIST LMDB data to Pytorch Tensor")

        for _k, (_dataimg, _label) in enumerate(_dataset):
            _data = _dataimg.astype(np.float32)
            self._normalize.fit(_data)
            _data =  self._normalize.transform(_dataimg)
            _data = _data.astype(np.float32)
            _data = self._transform(_data)
            _data = _data.view(1, _imgch, _imgheight, _imgwidth)

            self._imglist.append(_data)
            self._labellist.append(_label)

        print("Construction of MNIST Data Set is completed")

    def __len__(self):
        return len(self._labellist)

    def __getitem__(self, _idx):
        sample = {'image': self._imglist[_idx], 'label': self._labellist[_idx]}
        return sample

# =================================================================
# LMDB Data Set
# =================================================================
class   torch_service:
    def __init__(self, _description):
        self._dataset   = []

        self._imgch     = 0
        self._imgwidth  = 0
        self._imgheight = 0
        self._numData   = 0

        self.l_caffe_root   = []
        self._platform      = None
        self._os_id         = None
        self.caffe_root     = None
        self._caffe_python_path = None
        self.caffe_obj      = None
        self.caffe_pb2      = None

        self.args   = self._ParseArgument(_description)
        self.caffe_obj, self.caffe_pb2 = self._caffe_import(_GLOGLevel='3')

    def _caffe_import(self, _GLOGLevel='2'):
        # -------------------------------------------------------------------------
        # _os_id :: Windows : 0 and Linux : 1
        # Platform Name ('Windows or Not)
        # -------------------------------------------------------------------------
        self.l_caffe_root.append('c:\\Projects\\caffe\\')
        self.l_caffe_root.append('/home/sderoen/caffe/')

        self._platform      = platform.system()
        self._os_id         = (lambda _c: 0 if _c == 'Windows' else 1)(self._platform)
        self.caffe_root     = self.l_caffe_root[self._os_id]
        self._caffe_python_path = os.path.join(self.caffe_root, 'python')

        os.environ['GLOG_minloglevel'] = _GLOGLevel
        sys.path.insert(0, self._caffe_python_path)
        try:
            import caffe
            from caffe.proto import caffe_pb2
            print("Import caffe : SUCCESS !!!", caffe)
        except:
            print("Import caffe : FAIL !!!")
            exit(0)

        return caffe, caffe_pb2


    def _ParseArgument(self, _description):
        parser = argparse.ArgumentParser(prog='torch_service.py',
                                         formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=textwrap.dedent(_description))

        parser.add_argument('-l', '--LMDB_dir',
                            help=" Indicate the directory of LMDB",
                            default="mnist_train_lmdb", type=str)

        parser.add_argument('-id', '--dataid',
                            help=" Data ID in LMDB",
                            default=0, type=int)

        args = parser.parse_args()

        return args

    # =================================================================
    # Small Service Function
    # =================================================================
    def _getLMDBdata(self, _LMDBpath):
        global lmdb_env
        self._dataset = []

        lmdb_file = _LMDBpath
        try:
            lmdb_env    = lmdb.open(lmdb_file)
            print("LMDB file Open Success!! Reading the data")
        except:
            print("LMDB file is not exists @ %s" %lmdb_file)
            exit(0)

        lmdb_txn    = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum       = self.caffe_pb2.Datum()

        return lmdb_txn, lmdb_cursor, datum

    # =================================================================
    # I/O Processing
    # =================================================================
    def _readLMDB(self, _LMDBpath):
        global lmdb_env
        self._dataset = []

        lmdb_file = _LMDBpath
        try:
            lmdb_env    = lmdb.open(lmdb_file)
            print("LMDB file Open Success!! Reading the data")
        except:
            print("LMDB file is not exists @ %s" %lmdb_file)
            exit(0)

        lmdb_txn    = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum       = self.caffe_pb2.Datum()

        for key, value in lmdb_cursor:
            datum.ParseFromString(value)

            label   = datum.label
            data    = self.caffe_obj.io.datum_to_array(datum)
            imdata  = data.astype(np.uint8)                 # original (dim, col, row)
            imdata  = np.transpose(imdata, (1, 2, 0))       # Height(col), width(row) channel(dim)
            imdata  = imdata.reshape(-1, imdata.shape[1])

            _data_tuple = (imdata, label)
            self._dataset.append(_data_tuple)

        self._imgch     = datum.channels
        self._imgwidth  = datum.width
        self._imgheight = datum.height
        self._numData   = len(self._dataset)

        return self._dataset


# =================================================================
# Test Processing
# =================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("%s : %s " %(__file__, __name__))

    ts = torch_service(_description)
    _dataset = ts._readLMDB(ts.args.LMDB_dir)
    _dataidx = ts.args.dataid
    (im, _label) = _dataset[_dataidx]

    print("Number of data : %d" %len(_dataset))
    print("label ", _label)
    print("Data  ", np.shape(im))

    #plt.imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    #plt.show()

    print("Load MNIST Data to DataLoader provided by torch")
    train_data = MNISTDataSet(ts._imgch, ts._imgheight, ts._imgwidth, _dataset, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4)

    dataiter = iter(trainloader)
    _item = dataiter.next()

    _label = _item['label']
    print(_label)

    print("Test is finished")
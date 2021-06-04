#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Deep Neural Network Test (For LMDB test)
#
# Base URL     :
# 2020 09 07 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
LMDB_test.py
                    Written by Jinwuk @ 2020-09-07
====================================================
Example : python LMDB_test.py 
'''
import numpy as np
from torch_service import torch_service, MNISTDataSet
import matplotlib.pyplot as plt

ts = torch_service(_description)
lmdb_txn, lmdb_cursor, datum = ts._getLMDBdata(ts.args.LMDB_dir)

_k, _limit = 0, 9
print("=======================================================")
print("LMDB file : ", ts.args.LMDB_dir)

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label       = datum.label
    _imgch      = datum.channels
    _imgwidth   = datum.width
    _imgheight  = datum.height

    data = ts.caffe_obj.io.datum_to_array(datum)
    imdata = data.astype(np.uint8)  # original (dim, col, row)
    imdata = np.transpose(imdata, (1, 2, 0))  # Height(col), width(row) channel(dim)
    imdata = imdata.reshape(-1, imdata.shape[1])

    print("Data %3d Label: %2d Ch: %2d (%3d x %3d) " %(_k, label, _imgch, _imgwidth, _imgheight))
    _k += 1

    if _k > _limit:
        break

print("=======================================================")
print("Process Finished ")
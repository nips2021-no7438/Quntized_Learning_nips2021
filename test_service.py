#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - Service Function
#
# 2021 03 09 by ***********
###########################################################################
_description = '''\
====================================================
torch_service.py
                    Written by *********** @ 2020-07-03
====================================================
only for the torch based test bench 
'''

import pickle
import torch

class random_data :
    def __init__(self, filename):
        self._first_phase = True
        self._file_name   = filename

    def _save_data(self, _data):
        with open(self._file_name, 'wb') as f:
            pickle.dump(_data, f, pickle.HIGHEST_PROTOCOL)

    def _load_data(self):
        with open(self._file_name, 'rb') as f:
            _data = pickle.load(f)
        return _data

# =================================================================
# Test Processing
# =================================================================
if __name__ == "__main__":
    # import the necessary packages
    import os

    print("-------------------------------------------------------")
    print(" 0. Data Generation")
    print("-------------------------------------------------------")
    print(" Working Path : ", os.getcwd())

    # System Data
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # GPU에서 실행하려면 이 주석을 제거하세요.

    N, D_in, H, D_out = 64, 1000, 100, 10

    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    _data = [x, y]


    print("-------------------------------------------------------")
    print(" 1. File Processing")
    print("-------------------------------------------------------")
    # Set the procedure
    _file_name = 'random_data.pickle'

    # Check the data
    print("File Name : ", _file_name)

    #Save the data
    if os.path.isfile(_file_name):
        print("File %s is already exist !! No SAVE" %_file_name)
    else:
        dt_module = random_data(_file_name)
        dt_module._save_data(_data)
        print("File %s is generated as pickle file" %_file_name)

    print("-------------------------------------------------------")
    print(" 2. Finish the Processing")
    print("-------------------------------------------------------")

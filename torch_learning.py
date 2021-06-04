#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# Pytorch - My manual Learning Algorithm based on Pytorch
# Working Directory : D:\Work_2021\python_test_2021\nips_2021_test\
#
# 2020 07 02 by Jinseuk Seok
###########################################################################
_description = '''\
====================================================
torch_nn02.py : Based on torch module
                    Written by Jinwuk @ 2021-03-10
====================================================
Example : python torch_nn02.py
'''
#-------------------------------------------------------------
# Description of Optimizer
#-------------------------------------------------------------
import math
import torch
from typing import Callable, Iterable, Optional, Tuple, Union
from torch.optim import Optimizer

import my_debug as DBG
#-------------------------------------------------------------
# Quantization based on pytorch
#-------------------------------------------------------------
class Quantization:
    def __init__(self):
        self.Q_base = 2
        self.Q_eta = 1
        self.Q_index = 0
        self.Q_param = self.eval_QP(self.Q_index)

    def eval_QP(self, index):
        _index = index if index > -1 else 0
        _res = self.Q_eta * pow(self.Q_base, _index)
        return _res

    def set_QP(self, index=-1):
        _index = self.Q_index if index == -1 else index
        self.Q_param = self.eval_QP(_index)

    def get_QP(self):
        return self.Q_param

    def get_Quantization(self, X):
        _X = X if torch.is_tensor(X) else torch.from_numpy(X)
        _X1 = self.Q_param * _X + 0.5
        _X2 = torch.floor(_X1)
        _res = (1.0 / self.Q_param) * _X2

        return _res

class Temperature:
    def __init__(self, base, index):
        self._C = pow(base, index)  # Hyper parameter
        self._beta = 20.0  # Speed control
        self._dim = 64  # Dimension of input/data
        self._eta = 1  # must be 1

    def inf_sigma(self, t):
        _res = self._C / torch.log(t + 2)
        return _res

    def T(self, t):
        _pow = self._beta / (t + 2)
        _res = torch.pow(2, 2.0 * _pow) * self.inf_sigma(t)
        return _res

    def obj_function(self, x, _func):
        _gamma  = self._dim / (24.0 * pow(self._eta, 2))
        _res    = 0.5 * torch.log2(_gamma / _func(x))
        return _res

    #-------------------------------------------------------------
    # Quantization
    #-------------------------------------------------------------
class Q_process:
    def __init__(self, bQuantization, index_limit, QuantMethod = 1):
        self.bQuantization  = bQuantization
        self.index_limit    = index_limit
        self.QuantMethod    = QuantMethod        # 1 or 2, True ....or delete, For debug
        self.c_qtz          = Quantization()
        self.c_tmp          = Temperature(base=10, index=-6)
        self.l_index_trend  = []
        self.l_infindextrend= []
        self.l_index_layer  = []

    def Quantization(self, x):
        if self.bQuantization:
            Xq = self.c_qtz.get_Quantization(x)
        else:
            Xq = x
        return Xq, x

    def Increment_Index(self, _Xq, _Xf, _index):
        _res = _Xq
        while True:
            if _res.sum() == 0:
                if _index > self.index_limit:
                    break
                else:
                    _index += 1
                    self.c_qtz.set_QP(index=_index)
                    _res = self.c_qtz.get_Quantization(_Xf)
            else:
                break

        return _res, _index

    def Limited_Index(self, step, _index, _infindex):
        if self.QuantMethod > 1:
            _infindex = self.c_tmp.obj_function(step, self.c_tmp.T)
            while True:
                if _infindex > _index:
                    _index += 1
                else:
                    break
        else:
            pass
        return _index, _infindex

    def Adv_quantize(self, Xq, Xf, step):
        _res = Xq
        if self.bQuantization and self.QuantMethod > 0:
            _index      = self.l_index_trend[-1]
            _infindex   = self.l_infindextrend[-1]
            if step == 0:
                self.l_index_layer.append(_index)
            else:
                if self.QuantMethod > 0:
                    _res, _index        = self.Increment_Index(_res, Xf, _index)
                    _index, _infindex   = self.Limited_Index(step, _index, _infindex)
                else:
                    pass
            self.l_index_trend.append(_index)
            self.l_infindextrend.append(_infindex)
        else:
            pass

        return _res

    def Get_QPIndex(self):
        _res = self.l_index_trend[-1]
        return _res

    ## For degug and Development
    def Get_lengthofIndexTrend(self):
        _res = len(self.l_index_trend)
        return _res

    def Get_InfQPIndex(self):
        _res = self.l_infindextrend[-1]
        return _res

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 1. AdamW
#-------------------------------------------------------------
class AdamW(Optimizer):
    """
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Original
                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                # Alternative
                hc      = torch.div(exp_avg, denom)
                hf      = torch.mul(hc, -step_size)
                p.data.add_(hf)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtAdamW
#-------------------------------------------------------------
class QtAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        Qparam=0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)

        # Qunatrization
        self.Q_proc = Q_process(bQuantization= True, index_limit= 32)
        self.Q_proc.c_qtz.set_QP(index=Qparam)
        self.Q_proc.l_index_trend.append(Qparam)
        self.Q_proc.l_infindextrend.append(Qparam)

        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                #p.data.addcdiv_(exp_avg, denom, value=-step_size)
                hc      = torch.div(exp_avg, denom)
                h       = torch.mul(hc, -step_size)
                # Quantization
                hq,_    = self.Q_proc.Quantization(h)
                hq      = self.Q_proc.Adv_quantize(hq, h, state["step"])
                p.data.add_(hq)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#-------------------------------------------------------------
# Learning Algorithm based on Pytorch 2. QtSGD
#-------------------------------------------------------------
class QtSGD(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float   = 1e-3,
            momentum    = 0, dampening   = 0, weight_decay= 0, Qparam=0, nesterov    = False
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, Qparam=Qparam, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Qunatrization
        self.Q_proc = Q_process(bQuantization= True, index_limit= 32)
        self.Q_proc.c_qtz.set_QP(index=Qparam)
        self.Q_proc.l_index_trend.append(Qparam)
        self.Q_proc.l_infindextrend.append(Qparam)
        self.Q_proc.l_index_layer.append(Qparam)

        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            step_size = group['lr']
            for _k, p in enumerate(group["params"]):
                # get Grad and state
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization and update
                state["step"] = 0 if len(state) == 0 else (state["step"] + 1)

                # For Debug
                if state["step"] == 1:
                    print("Start Debug")

                # get Directional Derivation
                h = -step_size * grad

                # Qunatization
                hq,_ = self.Q_proc.Quantization(h)
                hq   = self.Q_proc.Adv_quantize(hq, h, state["step"])

                # Update weight by Learning equation #w_t - step_size * g_t
                p.data.add_(hq, alpha=1.0)

        return loss

#-------------------------------------------------------------
# Learning Module (top)
#-------------------------------------------------------------
class learning_module:
    def __init__(self, model, args):
        self.model      = model
        self.args       = args
        self.optimizer  = self.set_optimizer()
        self.epoch      = 0

        print("-----------------------------------------------------------------")
        print("model \n ", self.model)
        print("-----------------------------------------------------------------")

    def set_optimizer(self):
        _parameters    = self.model.parameters()
        _learning_rate = self.args.learning_rate

        if self.args.model_name == 'Adam':
            _optimizer = torch.optim.Adam(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'QSGD':
            _optimizer = QtSGD(_parameters, lr=_learning_rate, Qparam=self.args.QParam)
        elif self.args.model_name == 'AdamW':
            _optimizer = AdamW(_parameters, lr=_learning_rate)
        elif self.args.model_name == 'QtAdamW':
            _optimizer = QtAdamW(_parameters, lr=_learning_rate, Qparam=self.args.QParam)
        else:
            _optimizer = torch.optim.SGD(_parameters, lr=_learning_rate)
        return _optimizer

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def learning(self, epoch):
        self.epoch = epoch
        self.optimizer.step()

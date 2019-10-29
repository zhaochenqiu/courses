import torch.nn.functional as F
import torch

from torch.autograd import gradcheck
from torch.autograd import Function

from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from scipy.linalg import expm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Exp(Function):

    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors

        return grad_output * result


class Exp2(Function):

    @staticmethod
    def forward(ctx, i):
        i = i * 2
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors

        return grad_output * result * 2




class MatrixExp(Function):

    @staticmethod
    def forward(ctx, C):

        A = torch.eye(3).double()

        H = A
        for i in torch.arange(1, 10000):
            A = torch.mm(A/i, C)
            H = H + A


        ctx.save_for_backward(H)

        return H

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors

        return torch.mm(grad_output, result)






def MatExp(C):
    A = torch.eye(3).to(device)

    H = A
    for i in torch.arange(1,10):
        A = torch.mm(A/i,C)
        H = H + A

#    print(expm(C.cpu().detach().numpy()))
    return H


class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None






#input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)

input = Variable(torch.randn(3, 3).double(), requires_grad=True)
test = gradcheck(MatrixExp.apply, input)
print(test)

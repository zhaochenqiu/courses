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

        return grad_output




class MatrixExp(Function):

    @staticmethod
    def forward(ctx, C):

        A = torch.eye(3).to(device)

        H = A
        for i in torch.arange(1, 10):
            A = torch.mm(A/i, C)
            H = H + A


        ctx.save_for_backward(H)
#        result = i.exp()
#        ctx.save_for_backward(result)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors

        print("result", result)
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



# linear = Exp.apply
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias



input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(LinearFunction.apply, input)
print(test)

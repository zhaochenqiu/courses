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

        print("result:", result)
        print("grad_output:", grad_output)
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

        print("grad_output:", grad_output)
        print("result:", result)
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






linear2 = Exp2.apply



learning_rate = 1e-2


v = Variable(torch.zeros(1, 1, 1).to(device), requires_grad = True  )
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

# for i in range(100):
#
#     a = v * 2
#     b = torch.exp(a)
#
#     loss = torch.abs(2 - b)
#
#     print(loss)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()



for i in range(100):
    loss = torch.abs(2 - linear2(v))


    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





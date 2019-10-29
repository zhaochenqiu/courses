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






# linear2 = Exp.apply
#
#
#
# learning_rate = 1e-2
#
#
# v = Variable(torch.zeros(1, 1, 1).to(device), requires_grad = True  )
# optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)
#
# for i in range(100):
#     loss = torch.abs(2 - linear2(v))
#
#
#     print(loss)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()




# print(input)
# print(output)


layer_matrixexp = MatrixExp.apply

B = torch.zeros(8,3,3).to(device)
B[0,0,2] = 1.0
B[1,1,2] = 1.0
B[2,0,1] = 1.0
B[3,1,0] = 1.0
B[4,0,0], B[4,1,1] = 1.0, -1.0
B[5,1,1], B[5,2,2] = -1.0, 1.0
B[6,2,0] = 1.0
B[7,2,1] = 1.0

learning_rate = 1e-5



init_tensor = torch.zeros(8, 1, 1)

init_tensor[0] += 0.01
init_tensor[1] += 0.02
init_tensor[2] += 0.03
init_tensor[3] += 0.04
init_tensor[4] += 0.05
init_tensor[5] += 0.06
init_tensor[6] += 0.07
init_tensor[7] += 0.08


# v = Variable((torch.zeros(8,1,1)).to(device), requires_grad=True)
v = Variable((init_tensor).to(device), requires_grad=True)

optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

H_t = torch.tensor([[ 1.0004,  0.0055,  0.0143],
        [-0.0051,  1.0000, -0.0017],
        [-0.0052,  0.0044,  0.9997]]).to(device)

#A = B*v
# C = torch.sum(B*v, 0)
#
# H1 = MatExp(C)
#
# print(H1)



for i in range(10000):

    C = torch.sum(B*v, 0)

    H = layer_matrixexp(C)
#    H = MatExp(C)

    loss = torch.sum( torch.abs(H - H_t))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("i:", i, "loss:", loss.cpu().detach().numpy())



import torch.nn.functional as F
import torch

from torch.autograd import gradcheck
from torch.autograd import Function

from torch.autograd import Variable
import torch.optim as optim


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

def MatrixExp(B, u):
    C = torch.sum(B*u,0)
    A = torch.eye(3).to(device)
    H = A + C
    for i in torch.arange(2,10):
        A = torch.mm(A/i,C)
        H = H + A
    return H


# linear2 = Exp.apply
#
#
# input = torch.randn(4, 4, dtype = torch.double, requires_grad = True)
# output = linear2(input)
#
# print(input)
# print(output)


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


v = Variable(torch.zeros(8,1,1).to(device), requires_grad=True)
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

H_t = torch.tensor([[ 1.0004,  0.0055,  0.0143],
        [-0.0051,  1.0000, -0.0017],
        [-0.0052,  0.0044,  0.9997]]).to(device)



for i in range(1000):

    H = MatrixExp(B, v)

    loss = torch.sum( torch.abs(H - H_t))


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)



print(H_t)

print(H)

print(loss)
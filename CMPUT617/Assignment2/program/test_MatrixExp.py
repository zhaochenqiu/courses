import torch.nn.functional as F
import torch

from torch.autograd import gradcheck
from torch.autograd import Function

from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from scipy.linalg import expm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def MatExp(B, u):
    C = torch.sum(B*u, 0)
    A = torch.eye(3).to(device)

    H = A
    for i in torch.arange(1,10):
        A = torch.mm(A/i,C)
        H = H + A

    return H


def MatrixExp(B, u):

  C = torch.sum(B*u,0)
  A = torch.eye(3).to(device)

  # after this step, the A should becomes A = torch.mm(A/i, C) instead of identity matrix
  H = A + C
  for i in torch.arange(2,10):
    A = torch.mm(A/i,C)
    H = H + A
  return H



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


C = torch.sum(B*v, 0)

H = MatrixExp(B, v)
H1 = MatExp(B, v)

print("The result of MatrixExp:", H)
print("The result of Modified MatrixExp:", H1)

print("The result of expm:", expm(C.cpu().detach().numpy()))


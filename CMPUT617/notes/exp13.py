from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable


X = torch.tensor([[1.0,0.0,0.0,1.0],[0.0,0.0,1.0,1.0]],dtype=torch.float32) # 2x4 matrix
X = torch.transpose(X,0,1)
Y = torch.tensor([[1.0,0.0,1.0,0.0]],dtype=torch.float32)                   # 1x4 vector
Y = torch.transpose(Y,0,1)
print("input: ", X)
print("output: ", Y)


W1 = Variable(torch.torch.FloatTensor(2, 8).uniform_(-1, 1), requires_grad=True) # 2x8 matrix
b1 = Variable(torch.zeros((1,8)), requires_grad=True)                            # 1x8 matrix
W2 = Variable(torch.torch.FloatTensor(8, 1).uniform_(-1, 1), requires_grad=True) # 8x1 matrix
b2 = Variable(torch.zeros([1]), requires_grad=True)


learning_rate = 0.05
optimizer = torch.optim.SGD([W1, b1, W2, b2], lr=learning_rate, momentum=0.9)    # Torch optimizer

loss_fn = torch.nn.MSELoss() # Eclidean loss function


Z1 = torch.mm(X,W1)    # 4x8 matrix
Z2 = Z1 + b1           # 4x8 matrix
Z3 = torch.sigmoid(Z2) # 4x8 matrix
Z4 = torch.mm(Z3,W2)   # 4x1 vector
Z5 = Z4 + b2           # 4x1 vector
Yp = torch.sigmoid(Z5) # 4x1 vector


print("W1:", W1, "b1:", b1, "W2:", W2, "b2:", b2)

optimizer.zero_grad()
loss = loss_fn(Yp, Y)
print("loss:", loss)
print("Yp:", Yp)
print("Y:", Y)
loss.backward()
print("loss:", loss)
print("Yp:", Yp)
print("Y:", Y)

optimizer.step()
print("W1:", W1, "b1:", b1, "W2:", W2, "b2:", b2)

print("lost:", loss.item())



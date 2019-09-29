from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable

X = torch.tensor([[1.0, 0.0, 0.0, 1.0],[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
X = torch.transpose(X, 0, 1)
Y = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
Y = torch.transpose(Y, 0, 1)


print("Input:", X)
print("Output:", Y)


W1 = Variable(torch.torch.FloatTensor(2, 8).uniform_(-1, 1), requires_grad=True)
b1 = Variable(torch.zeros((1, 8)), requires_grad=True)
W2 = Variable(torch.torch.FloatTensor(8, 1).uniform_(-1, 1), requires_grad=True)
b2 = Variable(torch.zeros([1]), requires_grad=True)

learning_rate = 0.5

for step in range(10000):

    Z1 = torch.mm(X, W1)
    Z2 = Z1 + b1
    Z3 = torch.sigmoid(Z2)
    Z4 = torch.mm(Z3, W2)
    Z5 = Z4 + b2
    Yp = torch.sigmoid(Z5)


    dYp = Yp - Y
    dZ5 = torch.sigmoid(Z5)*(1.0 - torch.sigmoid(Z5))*dYp
    dZ4 = dZ5
    dZ3 = torch.mm(dZ4, torch.transpose(W2, 0, 1))
    dZ2 = torch.sigmoid(Z2)*(1.0 - torch.sigmoid(Z2))*dZ3
    dZ1 = dZ2


    dW1 = torch.mm(torch.transpose(X, 0, 1), dZ1)
    db1 = torch.sum(dZ2, 0, True)
    dW2 = torch.mm(torch.transpose(Z3, 0, 1), dZ4)
    db2 = torch.sum(dZ5)


    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    if step%1000 == 0:
        loss = torch.sum((Yp - Y)**2)
        print("loss:", loss)

print(Yp)
print(Y)




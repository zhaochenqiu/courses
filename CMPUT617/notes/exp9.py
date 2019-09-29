import torch
from torch.autograd import Variable

x1 = Variable(torch.tensor([1.0, -1.0]), requires_grad=True)


def compute_x2(x1):
    x2 = torch.zeros(4)

    x2[0] = x1[0] - x1[1]
    x2[1] = x1[0]**2 + x1[1]
    x2[2] = 3*x1[0] + 2*x1[1]
    x2[3] = (2*x1[0] + x1[1])**2

    return x2


def compute_x3(x2):
    x3 = torch.zeros(3)

    x3[0] = 4*x2[0] + (x2[2] + x2[3])**2
    x3[1] = 5*x2[1] + x2[3]**3
    x3[2] = 2*x2[0] + (x2[1] + x2[2])**2 + x2[3]**4

    return x3


def compute_x4(x3):
    x4 = torch.zeros(2)

    x4[0] = 4*x3[0] + 2*x3[1] + x3[2]**2
    x4[1] = (2*x3[0] + x3[2])**2

    return x4

def compute_x5(x4):
    x5 = (x4[0] + x4[1])**3

    return x5




x2 = compute_x2(x1)
x3 = compute_x3(x2)
x4 = compute_x4(x3)
x5 = compute_x5(x4)


g = torch.autograd.grad(x5, x1)

print("g:", g)

#
#
# print("x2:", x2)
# print("x3:", x3)
# print("x4:", x4)
# print("x5:", x5)
#
#
# g_x5 = torch.autograd.grad(x5, x4, retain_graph=True, create_graph=True)
# g_x4 = torch.autograd.grad(x4, x3, g_x5, retain_graph=True, create_graph=True)
# g_x3 = torch.autograd.grad(x3, x2, g_x4, retain_graph=True)
# g_x2 = torch.autograd.grad(x2, x1, g_x3, retain_graph=True)
#
# print("Gradient g_x5:", g_x5[0])
# print("Gradient g_x4:", g_x4[0])
# print("Gradient g_x3:", g_x3[0])
# print("Gradient g_x2:", g_x2[0])

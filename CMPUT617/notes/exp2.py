import torch
from torch.autograd import Variable

x = Variable(torch.tensor([3.0, -1.0, 0.0, 1.0]), requires_grad=True)
f = (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] + 2*x[2])**4 + 10*(x[0] - x[3])**4

print("Vector variable x:", x.data)
print("Function f at x:", f.data)


# compute gradient of f at x
g = torch.autograd.grad(f, x)

print("Gradient of f at x:", g[0].data)
print("g:",g)

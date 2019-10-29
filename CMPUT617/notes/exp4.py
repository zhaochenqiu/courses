import torch
from torch.autograd import Variable

x = Variable(torch.tensor([3.0, -1.0, 0.0, 1.0]), requires_grad = True)

slen = 1e-3

for i in range(1000):
    f = (x[0]+10.0*x[1])**2 + 5.0*(x[2]-x[3])**2 + (x[1]-2.0*x[2])**4 + 10.0*(x[0]-x[3])**4
    g = torch.autograd.grad(f, x)
    x = x - slen*g[0]
    if i%100 == 0:
        # item() → number Returns the value of this tensor as a standard Python number. This only works for tensors with one element
        print("Current variable value:", x.detach().numpy(), "Current function value:", f.item())


print("before")
print(chr(27) + "[2J")
print("after")

x = Variable(torch.tensor([3.0, -1.0, 0.0, 1.0]), requires_grad = True)

optimizer = torch.optim.SGD([x], lr=1e-3, momentum=0.9)

for i in range(100):
    f = (x[0]+10.0*x[1])**2 + 5.0*(x[2]-x[3])**2 + (x[1]-2.0*x[2])**4 + 10.0*(x[0]-x[3])**4
    optimizer.zero_grad()
    f.backward()
    optimizer.step()
    if i%10==0:
        print("Current variable value:", x.detach().numpy(), "Current function value:", f.item())



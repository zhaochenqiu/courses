import torch
from torch.autograd import Variable

z = Variable(torch.tensor([1.0, -1.0]), requires_grad=True)

print("Variable z:", z)

def compute_x(z):
    x = torch.zeros(4)

    x[0] = z[0] - z[1]
    x[1] = z[0]**2
    x[2] = z[1]**2
    x[3] = z[0]**2+z[0]*z[1]

    return x


x = compute_x(z)
print("function x:", x)


def compute_f(x):
    f = (x[0]+10.0*x[1])**2 + 5.0*(x[2]-x[3])**2 + (x[1]-2.0*x[2])**4 + 10.0*(x[0]-x[3])**4

    return f

f = compute_f(x)
print("function f:", f)
print("")


g_x = torch.autograd.grad(f,x,retain_graph=True,create_graph=True)
g_z = torch.autograd.grad(x,z,g_x,retain_graph=True)

g = torch.autograd.grad(f,z)

print("Gradient by chain rule:",g_z[0])
print("Gradient by PyTorch:",g[0])


steplength = 1e-3
for i in range(1000):
    f = compute_f(compute_x(z))
    g = torch.autograd.grad(f, z)

    z = z - steplength*g[0]

    if i%100 == 0:
        print("Current variable value:", z.detach().numpy(), "Current function value:", f.item())


z = Variable(torch.tensor([1.0, -1.0]), requires_grad=True)

optimizer = torch.optim.SGD([z], lr=1e-3, momentum=0.9)

for i in range(100):
    f = compute_f(compute_x(z))
    optimizer.zero_grad()
    f.backward()
    optimizer.step()
    if i%10 == 0:
        print("Current variable value:",z.detach().numpy(),"Current function value:", f.item())

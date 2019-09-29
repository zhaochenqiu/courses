import torch
from torch.autograd import Variable

x = Variable(torch.tensor([3.0, -1.0, 0.0, 1.0]), requires_grad=True)
f = (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] + 2*x[2])**4 + 10*(x[0] - x[3])**4

g = torch.autograd.grad(f, x, retain_graph=True, create_graph=True)




eye = torch.eye(4)

H = torch.stack([torch.autograd.grad(g,x,eye[:,i],retain_graph=True)[0] for i in range(4)]) # hessian


y = torch.tensor([1, 0, 0, 0])
print("y:", y)

H1 = torch.autograd.grad(g, x, eye[:, 1], retain_graph=True)


print("eye:", eye)
print("g:", g)

print("Hessian:",H.data)
print("H1:", H1)


x = Variable(torch.tensor([3.0,-1.0,0.0,1.0]),requires_grad=True)

steplength = 1e-3 # for gradient descent
for i in range(1000):
  # function
  f = (x[0]+10.0*x[1])**2 + 5.0*(x[2]-x[3])**2 + (x[1]-2.0*x[2])**4 + 10.0*(x[0]-x[3])**4
  # compute grdaient
  g = torch.autograd.grad(f,x)
  # adjust variable
  x = x - steplength*g[0]
  if i%100==0:
    print("Current variable value:",x.detach().numpy(),"Current function value:", f.item())

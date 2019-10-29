import torch
from torch.autograd import Variable

# RuntimeError: Only Tensors of floating point dtype can require gradients
# Must be 1.0 rather than 1
x = Variable(torch.tensor([1.0, -1.0]), requires_grad=True)

print(x)

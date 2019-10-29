from skimage import io

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim

from skimage.transform import pyramid_gaussian
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from torch.autograd import function

# from google.colab import drive

# drive.mount('/content/drivev')




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


I = io.imread('../fixed.bmp').astype(np.float32)
J = io.imread('../moving.bmp').astype(np.float32)


pyramid_I = tuple(pyramid_gaussian(I, downscale=2, multichannel=False))
pyramid_J = tuple(pyramid_gaussian(J, downscale=2, multichannel=False))



def PerspectiveTransform(I, H, xv, yv):

  # apply homography
  xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
  yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
  J = F.grid_sample(I.view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
  return J

def MatrixExp(B, u):

  C = torch.sum(B*u,0)
  A = torch.eye(3).to(device)
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

# multi-resolution registration
# Consider 8 levels in the pyramid for this example
learning_rate = 1e-5
nItr = torch.tensor([300,300,300,300,400,500,600,600])

torch.autograd.set_detect_anomaly(True)

# create variables and optimization at each level
v = Variable(torch.zeros(8,1,1).to(device), requires_grad=True)
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

#for level in torch.arange(7,-1,-1): # start at level 7

level = 7
I = torch.tensor(pyramid_I[level].astype(np.float32)).to(device)
J = torch.tensor(pyramid_J[level].astype(np.float32)).to(device)



height,width = I.shape

# choose a set of pixel locations on the template image that are most informative
tval = 0.9*threshold_otsu(I.cpu().numpy()) # reduce Otsu threshold value a bit to cover slightly wider areas
important_ind = torch.nonzero((I.data>tval).view([height*width])).squeeze()

# generate grid only once at each level
yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
# map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
yv = 2.0*yv/(height-1) - 1.0
xv = 2.0*xv/(width-1) - 1.0

for itr in range(2000):
  J_w = PerspectiveTransform(J, MatrixExp(B,v), xv, yv)
  loss = F.mse_loss(J_w.view([height*width])[important_ind], I.view([height*width])[important_ind])


  optimizer.zero_grad()

  loss.backward()
  optimizer.step()


H = MatrixExp(B,v).detach()
J_w = PerspectiveTransform(J, H, xv, yv)
loss = F.mse_loss(J_w, I)
print("Pyramid level:",level,"Iteration:",itr+1,"MSE loss:",loss.item())







# final transformation
I = torch.tensor(pyramid_I[0].astype(np.float32)).to(device) # without Gaussian
J = torch.tensor(pyramid_J[0].astype(np.float32)).to(device) # without Gaussian


height, width = I.shape

yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])

yv = 2.0*yv/(height-1) - 1.0
xv = 2.0*xv/(width-1) - 1.0

J_w = PerspectiveTransform(J, H, xv, yv)

D = J - I
D_w = J_w - I
print(" ")
print("MSE before registration:",torch.mean(D**2).cpu().item())
print("MSE after registration:",torch.mean(D_w**2).cpu().item())



print(H)
# %matplotlib inline
fig=plt.figure(figsize=(30,30))
fig.add_subplot(1,2,1)
plt.imshow(D.cpu().data)
plt.title("Difference image before registration")
fig.add_subplot(1,2,2)
plt.imshow(D_w.cpu().data)
plt.title("Difference image after registration")


plt.show()

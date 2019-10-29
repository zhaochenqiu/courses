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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


I = io.imread('../fixed.bmp').astype(np.float32)
J = io.imread('../moving.bmp').astype(np.float32)

I = torch.tensor(I).to(device)
J = torch.tensor(J).to(device)


def PerspectiveTransform(I, H, xv, yv):

  # apply homography
  print("xv in PerspectiveTransform:")
  print(xv.shape)

  print("H[0,0]")
  print(H[0,0])

  print(xv)
  xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
  yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
  print(xvt)

  temp = I.view(1, 1, height, width)
  temp2 = torch.stack([xvt, yvt], 2).unsqueeze(0)

  print("temp2:")
  print(temp2)
  print(torch.max(temp2))
  print("\n\n\n")
  print("temp.size():", temp.size())
  print("temp2.size():", temp2.size())


  temp3 = F.grid_sample(temp, temp2)

  print("temp3.size():", temp3.size())

  print("\n\n\n")



  J = F.grid_sample(I.view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
  return J



height,width = I.shape

H = torch.tensor([[ 1.0004,  0.0051,  0.0143],
        [-0.0050,  0.9999, -0.0018],
        [-0.0041,  0.0038,  0.9997]], device='cuda:0')

H = torch.tensor([[ 0.0,  -1.0,  0.0],
                   [1.0,   0.0,  0.0],
                   [0.0,   0.0,  1.0]], device='cuda:0')

H = torch.tensor([[ 1.0,   0.0,  1.0],
                   [0.0,   1.0,  1.0],
                   [0.0,   0.0,  1.0]], device='cuda:0')




yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
# map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
yv = 2.0*yv/(height-1) - 1.0
xv = 2.0*xv/(width-1) - 1.0




J_w = PerspectiveTransform(J, H, xv, yv)

print("xv:", xv.shape)
print("yv:", yv.shape)

print("J.shape:", J.shape)
print("J_w.shape:", J_w.shape)




fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(J.cpu().data)
plt.subplot(1,2,2)
plt.imshow(J_w.cpu().data)
plt.show()


# D = J - I
# D_w = J_w - I
#
#
# print(H)
# # %matplotlib inline
# fig=plt.figure(figsize=(30,30))
# fig.add_subplot(1,2,1)
# plt.imshow(D.cpu().data)
# plt.title("Difference image before registration")
# fig.add_subplot(1,2,2)
# plt.imshow(D_w.cpu().data)
# plt.title("Difference image after registration")
#
#
# plt.show()

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

fig1 = plt.figure()
plt.subplot(1,2,1)
plt.imshow(I)
plt.subplot(1,2,2)
plt.imshow(J)


pyramid_I = tuple(pyramid_gaussian(I, downscale=2, multichannel=False))
pyramid_J = tuple(pyramid_gaussian(J, downscale=2, multichannel=False))

fig2 = plt.figure()
fig2.add_subplot(1,2,1)
plt.imshow(pyramid_I[7])
plt.title("Fixed Image")
fig2.add_subplot(1,2,2)
plt.imshow(pyramid_J[7])
plt.title("Moving Image")

# plt.show()


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

print(B)



# multi-resolution registration
# Consider 8 levels in the pyramid for this example
learning_rate = 1e-5
nItr = torch.tensor([300,300,300,300,400,500,600,600])

torch.autograd.set_detect_anomaly(True)

# create variables and optimization at each level
v = Variable(torch.zeros(8,1,1).to(device), requires_grad=True)
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

for level in torch.arange(7,-1,-1): # start at level 7
  if level>0:
    I = torch.tensor(pyramid_I[level].astype(np.float32)).to(device)
    J = torch.tensor(pyramid_J[level].astype(np.float32)).to(device)

    print('level1 = ', level)
  else:

    # 做了一次高斯模糊
    I = torch.tensor(gaussian(pyramid_I[level].astype(np.float32),2.0)).to(device)
    J = torch.tensor(gaussian(pyramid_J[level].astype(np.float32),2.0)).to(device)

    print('level2 = ', level)

  height,width = I.shape

  # choose a set of pixel locations on the template image that are most informative

  # tval 返回的是一个阈值
  tval = 0.9*threshold_otsu(I.cpu().numpy()) # reduce Otsu threshold value a bit to cover slightly wider areas
  print('tval:', tval)

  print(I.data)
  print(I.shape)
  print('height, width', height, width)

  temp = (I.data>tval)
  print("temp:")
  print(temp)

  temp1 = temp.view([height*width])

  print("temp1:")
  print(temp1)
  print(temp1.size())




  important_ind = torch.nonzero((I.data>tval).view([height*width])).squeeze()
  print("important:")
  print(important_ind)



  # generate grid only once at each level
  yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
  # map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
  yv = 2.0*yv/(height-1) - 1.0
  xv = 2.0*xv/(width-1) - 1.0

  for itr in range(nItr[level]):
    J_w = PerspectiveTransform(J, MatrixExp(B,v), xv, yv)
    loss = F.mse_loss(J_w.view([height*width])[important_ind], I.view([height*width])[important_ind])
    #loss = F.mse_loss(J_w, I)
    optimizer.zero_grad()
    loss.backward()
    if itr%100==0:
      loss = F.mse_loss(J_w, I)
      print("Pyramid level:",level.item(),"Iteration:",itr,"MSE loss:",loss.item())
    optimizer.step()

  H = MatrixExp(B,v).detach()
  J_w = PerspectiveTransform(J, H, xv, yv)
  loss = F.mse_loss(J_w, I)
  print("Pyramid level:",level.item(),"Iteration:",itr+1,"MSE loss:",loss.item())

# final transformation
I = torch.tensor(pyramid_I[0].astype(np.float32)).to(device) # without Gaussian
J = torch.tensor(pyramid_J[0].astype(np.float32)).to(device) # without Gaussian
J_w = PerspectiveTransform(J, H, xv, yv)

D = J - I
D_w = J_w - I
print(" ")
print("MSE before registration:",torch.mean(D**2).cpu().item())
print("MSE after registration:",torch.mean(D_w**2).cpu().item())




def histogram_mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(), image2.ravel(), bins=100)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

print("MI before registration:", histogram_mutual_information(pyramid_I[0],pyramid_J[0]))
print("MI after registration:", histogram_mutual_information(pyramid_I[0],J_w.cpu().numpy()))

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
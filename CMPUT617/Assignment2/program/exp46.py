from skimage import io

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import skimage

from torch.autograd import Variable
import torch.optim as optim


from skimage.transform import pyramid_gaussian
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from torch.autograd import function
from torch.autograd import Function


import datetime

import pickle
import torch.autograd as autograd
import torch.nn as nn

# from google.colab import drive

# drive.mount('/content/drivev')

class MatrixExp(Function):

    @staticmethod
    def forward(ctx, C):

        A = torch.eye(3).to(device)

        H = A
        for i in torch.arange(1, 100):
            A = torch.mm(A/i, C)
            H = H + A

        ctx.save_for_backward(H)

        return H

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors

        return torch.mm(grad_output, result)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


I = io.imread('../fixed.bmp').astype(np.float32)
J = io.imread('../moving.bmp').astype(np.float32)
#

I = io.imread('../image_fixed.jpg')
J = io.imread('../image_moving4.jpg')

fig1 = plt.figure()
plt.subplot(1,2,1)
plt.imshow(I)
plt.subplot(1,2,2)
plt.imshow(J)
# plt.show()

# pyramid_I = tuple(pyramid_gaussian(I, downscale=2, multichannel=False))
# pyramid_J = tuple(pyramid_gaussian(J, downscale=2, multichannel=False))

pyramid_I = tuple(pyramid_gaussian(I, downscale=2, multichannel=True))
pyramid_J = tuple(pyramid_gaussian(J, downscale=2, multichannel=True))



# showim1 = pyramid_I[1]
# showim2 = pyramid_I[2]
# showim3 = pyramid_I[3]
#
#
#
# fig2 = plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(showim1)
# plt.subplot(1, 3, 2)
# plt.imshow(showim2)
# plt.subplot(1, 3, 3)
# plt.imshow(showim3)
#
# plt.show()

# pyramid_I[level].astype(np.float32)


# fig2 = plt.figure()
# fig2.add_subplot(1,2,1)
# plt.imshow(pyramid_I[7])
# plt.title("Fixed Image")
# fig2.add_subplot(1,2,2)
# plt.imshow(pyramid_J[7])
# plt.title("Moving Image")
#
# plt.show()


class Mine(nn.Module):
    def __init__(self, input_size=6, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output



def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))

    mi_lb = torch.mean(t) - torch.log(torch.mean(et))

    return mi_lb, t, et



def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]

#        print("joint batch:", batch.shape)
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)

        batch_joint = data[joint_index, 0:3]
        batch_marginal = data[marginal_index, 3:6]

        batch = torch.cat( (batch_joint, batch_marginal) , 1)
#
#         print("batch_joint:", batch_joint.shape)
#         print("batch_marginal:", batch_marginal.shape)
#         batch = torch.cat((data[joint_index][:,0:2].reshape(-1,1),
#                            data[marginal_index][:,3:5].reshape(-1,1)), 1)
#
#         print("marginal batch:", batch.shape)
    return batch

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

def MINE_loss(J_w, I, batch_size, important_ind):
    height, width, byte = I.shape

#     T = J_w.view(height*width, byte)
#     temp1 = T[important_ind, :]
#     print("important_ind", len(important_ind))
#     print("temp1:", temp1.shape)

    data_J = J_w.view(height*width, byte)
    data_J = data_J[important_ind]

    data_I = I.view(height*width, byte)
    data_I = data_I[important_ind]

    data = torch.cat( (data_J, data_I), 1)

#     print("data_J", data_J.shape)
#     print("data_I", data_I.shape)
#     print("data:", data.shape)
#
#     data = torch.cat(( J_w.view([height * width])[important_ind].unsqueeze(1), \
#                        I.view(  [height * width])[important_ind].unsqueeze(1)), 1)

    joint, marginal = sample_batch(data, batch_size), sample_batch(data, batch_size, 'marginal')

#    print("joint:", joint.shape)
#    print("marginal:", marginal.shape)
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))

#    print("t:", t.shape)
#    print("et:", et.shape)

    mine_loss = -(torch.mean(t) - torch.log(torch.mean(et)))

    return mine_loss


def PerspectiveTransform(I, H, xv, yv):

    height, width, byte = I.shape

    xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])

    J0 = F.grid_sample(I[:, :, 0].view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
    J1 = F.grid_sample(I[:, :, 1].view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
    J2 = F.grid_sample(I[:, :, 2].view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()

    J0 = J0.unsqueeze(2)
    J1 = J1.unsqueeze(2)
    J2 = J2.unsqueeze(2)


    J = torch.cat((J0, J1, J2), 2)


    return J

# def MatrixExp(B, u):
#     C = torch.sum(B*u,0)
#     A = torch.eye(3).to(device)
#     H = A + C
#     for i in torch.arange(2,10):
#         A = torch.mm(A/i,C)
#         H = H + A
#     return H

# def MatrixExp(B, u):
#     C = torch.sum(B*u,0)
#     A = torch.eye(3).to(device)
#
#     H = A
#     for i in torch.arange(1,10):
#         A = torch.mm(A/i,C)
#         H = H + A
#
#     return H


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
learning_rate = 1e-4
#nItr = torch.tensor([300,300,300,300,400,500,600,600])
nItr = torch.tensor([600,600,600,600,600,600,600,600])
nItr = torch.tensor([1000,1000,1000,1000,1000,1000,1000,1000])

nItr = torch.tensor([2000,2000,2000,2000,2000,2000,2000,2000])
nItr = torch.tensor([3000,3000,3000,3000,3000,3000,3000,3000])
nItr = torch.tensor([4000,4000,4000,4000,4000,4000,4000,4000])
nItr = torch.tensor([6000,6000,6000,6000,6000,6000,6000,6000])

nItr = torch.tensor([8000,8000,8000,8000,8000,8000,8000,8000])

nItr = torch.tensor([30000,30000,30000,30000,30000,30000,30000,30000])





list_batch = torch.tensor([12800, 6400, 3200, 1600 ,800, 400, 200 ,100])


# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)


#layer_matrixexp = MatrixExp.apply

# create variables and optimization at each level
v = Variable(torch.zeros(8,1,1).to(device), requires_grad=True)
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)
# optimizer = optim.SGD([v], lr=learning_rate)

# for level in torch.arange(7,6,-1): # start at level 7
mine_net = Mine().cuda()
mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)



starttime = datetime.datetime.now()



result_errors = list()
for level in torch.arange(4, -1, -1):
    if level > 0:
        I = torch.tensor(pyramid_I[level].astype(np.float32)).to(device)
        J = torch.tensor(pyramid_J[level].astype(np.float32)).to(device)
    else:
        I = torch.tensor(gaussian(pyramid_I[level].astype(np.float32),2.0)).to(device)
        J = torch.tensor(gaussian(pyramid_J[level].astype(np.float32),2.0)).to(device)


    height, width, byte = I.shape

#    print("In for I", I.shape)

    # choose a set of pixel locations on the template image that are most informative

    I_gray = torch.tensor( skimage.color.rgb2gray(I.cpu().numpy()) )
    tval = 0.1*threshold_otsu(I_gray.cpu().numpy()) # reduce Otsu threshold value a bit to cover slightly wider areas
    important_ind = torch.nonzero( (I_gray.data > tval).view([height*width]) ).squeeze()

    # generate grid only once at each level
    yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
    # map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
    yv = 2.0*yv/(height-1) - 1.0
    xv = 2.0*xv/(width-1) - 1.0



    for itr in range(nItr[level]):
#    for itr in range(1000):

        C = torch.sum(B*v, 0)
        J_w = PerspectiveTransform(J, MatrixExp.apply(C), xv, yv)

#         plt.figure()
#         plt.subplot(1, 2, 1)
#         plt.imshow(I.cpu().data)
#         plt.subplot(1, 2, 2)
#         plt.imshow(J_w.cpu().data)
#         plt.show()
#
#
#
#         print("J_w:", J_w.shape)

        mine_loss = MINE_loss(J_w, I, list_batch[level].detach().numpy(), important_ind)

        mine_net_optim.zero_grad()
        optimizer.zero_grad()

        mine_loss.backward(retain_graph=True)

        optimizer.step()
        mine_net_optim.step()

        if itr%100==0:
#            print("Iteration:",itr,"loss:",mine_loss)
            print("Pyramid level:",level.item(),"Iteration:",itr,"MINE loss:",mine_loss.item())


        result_errors.append( mine_loss.detach().cpu().numpy() )


    C = torch.sum(B*v, 0)
    H = MatrixExp.apply(C)
    J_w = PerspectiveTransform(J, H, xv, yv)
    mine_loss = MINE_loss(J_w, I, list_batch[level].detach().numpy(), important_ind)
    print("Pyramid level:",level.item(),"Iteration:",itr,"MINE loss:",mine_loss.item())


#    loss = F.mse_loss(J_w, I)



# final transformation
I = torch.tensor(pyramid_I[0].astype(np.float32)).to(device) # without Gaussian
J = torch.tensor(pyramid_J[0].astype(np.float32)).to(device) # without Gaussian

height, width, byte = I.shape

yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
# map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
yv = 2.0*yv/(height-1) - 1.0
xv = 2.0*xv/(width-1) - 1.0

J_w = PerspectiveTransform(J, H, xv, yv)

D = torch.abs(J - I)
D_w = torch.abs(J_w - I)

print(H)

fig=plt.figure(figsize=(30,30))
fig.add_subplot(1,2,1)
plt.imshow(D.cpu().data)
plt.title("Difference image before registration")
fig.add_subplot(1,2,2)
plt.imshow(D_w.cpu().data)
plt.title("Difference image after registration")





result_errors = ma(result_errors, 100)

plt.figure()
plt.plot(range(len(result_errors)), result_errors)


# with open('Matrix_layer.txt', 'w') as f:
#     f.write(str(result_errors))

with open('Matrix_layer.pkl', 'wb') as f:
    pickle.dump(result_errors, f)


endtime = datetime.datetime.now()

print("total time:", (endtime - starttime).seconds)


plt.show()
#
#
#

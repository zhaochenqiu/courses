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
from torch.autograd import Function


import torch.autograd as autograd
import torch.nn as nn

# from google.colab import drive

# drive.mount('/content/drivev')

class MatrixExp_Linear(Function):

    @staticmethod
    def forward(ctx, C):

        A = torch.eye(3).to(device)

        H = A
        for i in torch.arange(1, 10):
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

fig1 = plt.figure()
plt.subplot(1,2,1)
plt.imshow(I)
plt.subplot(1,2,2)
plt.imshow(J)


pyramid_I = tuple(pyramid_gaussian(I, downscale=2, multichannel=False))
pyramid_J = tuple(pyramid_gaussian(J, downscale=2, multichannel=False))

# fig2 = plt.figure()
# fig2.add_subplot(1,2,1)
# plt.imshow(pyramid_I[7])
# plt.title("Fixed Image")
# fig2.add_subplot(1,2,2)
# plt.imshow(pyramid_J[7])
# plt.title("Moving Image")

# plt.show()


class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
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
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)

        batch = torch.cat((data[joint_index][:,0].reshape(-1,1),
                                         data[marginal_index][:,1].reshape(-1,1)), 1)
    return batch

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

def MINE_loss(J_w, I, batch_size, important_ind):
    height, width = I.shape

    data = torch.cat(( J_w.view([height * width])[important_ind].unsqueeze(1), \
                       I.view(  [height * width])[important_ind].unsqueeze(1)), 1)

    joint, marginal = sample_batch(data, batch_size), sample_batch(data, batch_size, 'marginal')

    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))

    mine_loss = -(torch.mean(t) - torch.log(torch.mean(et)))

    return mine_loss


def PerspectiveTransform(I, H, xv, yv):
    xvt = (xv*H[0,0]+yv*H[0,1]+H[0,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    yvt = (xv*H[1,0]+yv*H[1,1]+H[1,2])/(xv*H[2,0]+yv*H[2,1]+H[2,2])
    J = F.grid_sample(I.view(1,1,height,width),torch.stack([xvt,yvt],2).unsqueeze(0)).squeeze()
    return J

# def MatrixExp(B, u):
#     C = torch.sum(B*u,0)
#     A = torch.eye(3).to(device)
#     H = A + C
#     for i in torch.arange(2,10):
#         A = torch.mm(A/i,C)
#         H = H + A
#     return H

def MatrixExp(B, u):
    C = torch.sum(B*u,0)
    A = torch.eye(3).to(device)

    H = A
    for i in torch.arange(1,10):
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

list_batch = torch.tensor([12800, 6400, 3200, 1600 ,800, 400, 200 ,100])


# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)


#layer_matrixexp = MatrixExp.apply

# create variables and optimization at each level
v = Variable(torch.zeros(8,1,1).to(device), requires_grad=True)
optimizer = optim.Adam([v], lr=learning_rate, amsgrad=True)

# for level in torch.arange(7,6,-1): # start at level 7

for level in torch.arange(7, -1, -1):
    if level > 0:
        I = torch.tensor(pyramid_I[level].astype(np.float32)).to(device)
        J = torch.tensor(pyramid_J[level].astype(np.float32)).to(device)
    else:
        I = torch.tensor(gaussian(pyramid_I[level].astype(np.float32),2.0)).to(device)
        J = torch.tensor(gaussian(pyramid_J[level].astype(np.float32),2.0)).to(device)


    height,width = I.shape

    # choose a set of pixel locations on the template image that are most informative
    tval = 0.9*threshold_otsu(I.cpu().numpy()) # reduce Otsu threshold value a bit to cover slightly wider areas
    important_ind = torch.nonzero((I.data>tval).view([height*width])).squeeze()

    # generate grid only once at each level
    yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
    # map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
    yv = 2.0*yv/(height-1) - 1.0
    xv = 2.0*xv/(width-1) - 1.0


    # result = train(rho_data,mine_net,mine_net_optim)
    mine_net = Mine().cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)


    result_errors = list()

    for itr in range(nItr[level]):
#    for itr in range(1000):

#        C = torch.sum(B*v, 0)
#        J_w = PerspectiveTransform(J, MatrixExp.apply(C), xv, yv)
        J_w = PerspectiveTransform(J, MatrixExp(B, v), xv, yv)


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


#    C = torch.sum(B*v, 0)
#    H = MatrixExp.apply(C)

    H = MatrixExp(B, v)
    J_w = PerspectiveTransform(J, H, xv, yv)
    mine_loss = MINE_loss(J_w, I, list_batch[level].detach().numpy(), important_ind)
    print("Pyramid level:",level.item(),"Iteration:",itr,"MINE loss:",mine_loss.item())


#    loss = F.mse_loss(J_w, I)



# final transformation
I = torch.tensor(pyramid_I[0].astype(np.float32)).to(device) # without Gaussian
J = torch.tensor(pyramid_J[0].astype(np.float32)).to(device) # without Gaussian

height, width = I.shape

yv, xv = torch.meshgrid([torch.arange(0,height).float().to(device), torch.arange(0,width).float().to(device)])
# map coordinates to [-1,1]x[-1,1] so that grid_sample works properly
yv = 2.0*yv/(height-1) - 1.0
xv = 2.0*xv/(width-1) - 1.0

J_w = PerspectiveTransform(J, H, xv, yv)

D = J - I
D_w = J_w - I

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



plt.show()

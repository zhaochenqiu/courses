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

import torch.autograd as autograd
import torch.nn as nn

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

        batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                         data[marginal_index][:,1].reshape(-1,1)],
                                       axis=1)
    return batch


def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    joint , marginal = batch
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()

    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()

    mi_lb , t, et = mutual_information(joint, marginal, mine_net)

    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)

    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size), sample_batch(data,batch_size=batch_size,sample_mode='marginal')

        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())

        if (i+1)%(log_freq)==0:
            print(result[-1])
    return result

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]





def PerspectiveTransform(I, H, xv, yv):
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

# for level in torch.arange(7,6,-1): # start at level 7


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


mine_net = Mine().cuda()
mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
# result = train(rho_data,mine_net,mine_net_optim)



for itr in range(1000):
    J_w = PerspectiveTransform(J, MatrixExp(B,v), xv, yv)

    input_J = J_w.view([height*width])[important_ind]
    input_I = I.view([height*width])[important_ind]

    temp_J = input_J.unsqueeze(1)
    temp_I = input_I.unsqueeze(1)

#     data = torch.cat((temp_J, temp_I), 1)
#     data = data.detach().cpu().numpy()
#     print("data.shape:", data.shape)
#     print("type:", type(data))
#
#     result = train(data, mine_net, mine_net_optim)
#     result_ma = ma(result)
#     print("result_ma:", result_ma[-1])


    loss = F.mse_loss(input_J, input_I)
#    print(loss)

    optimizer.zero_grad()
    loss.backward()

    if itr%100==0:
        loss = F.mse_loss(J_w, I)
        print("Iteration:",itr,"MSE loss:",loss.item())

    optimizer.step()



H = MatrixExp(B,v).detach()
J_w = PerspectiveTransform(J, H, xv, yv)
loss = F.mse_loss(J_w, I)



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



fig=plt.figure(figsize=(30,30))
fig.add_subplot(1,2,1)
plt.imshow(D.cpu().data)
plt.title("Difference image before registration")
fig.add_subplot(1,2,2)
plt.imshow(D_w.cpu().data)
plt.title("Difference image after registration")

plt.show()

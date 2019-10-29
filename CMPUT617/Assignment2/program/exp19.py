import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import bokeh
import numpy as np

import matplotlib.pyplot as plt



# data
var = 0.2
def func(x):
    return x



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

def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]




H=100
n_epoch = 5000

batch_size = 100




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(1, H)
        self.fc3 = nn.Linear(H, 1)
        self.fc4 = nn.Linear(H, H)

    def forward(self, x, y):
        h1 = F.elu(self.fc1(x)+self.fc2(y))
        h2 = F.elu(self.fc4(h1))
        h3 = self.fc3(h2)
        return h3

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
plot_loss = []

rho = 0.9
data = np.random.multivariate_normal( mean=[0,0],
                                cov=[[1,rho],[rho,1]],
                                size = 300)

# print(data.shape)

for epoch in tqdm(range(n_epoch)):

#    print(data.shape)

    joint, marginal = sample_batch(data,batch_size=batch_size), sample_batch(data,batch_size=batch_size,sample_mode='marginal')


    sample_x = joint[:, 0]
    sample_y = joint[:, 1]

    shuffle_x = marginal[:, 0]
    shuffle_y = marginal[:, 1]


    sample_x = sample_x.reshape(len(sample_x), 1)
    sample_y = sample_y.reshape(len(sample_y), 1)

    shuffle_x = shuffle_x.reshape(len(shuffle_x), 1)
    shuffle_y = shuffle_y.reshape(len(shuffle_y), 1)


    x_sample = sample_x
    y_sample = sample_y
    x_shuffle = shuffle_x
    y_shuffle = shuffle_y




    x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad = True)
    y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad = True)

    x_shuffle = Variable(torch.from_numpy(x_shuffle).type(torch.FloatTensor), requires_grad = True)
    y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad = True)



    pred_xy = model(x_sample, y_sample)
    pred_x_y = model(x_shuffle, y_shuffle)

    ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
    loss = - ret  # maximize
    plot_loss.append(loss.data.numpy())
    model.zero_grad()
    loss.backward()
    optimizer.step()


print(x_sample)



plot_x = np.arange(len(plot_loss))
plot_y = np.array(plot_loss).reshape(-1,)



plot_ma_y = ma(-plot_y)



plt.figure()
plt.plot(plot_x,- plot_y)

plt.figure()
plt.plot(range(len(plot_ma_y)), plot_ma_y)

print(len(plot_ma_y))

print(-plot_y)
print(plot_ma_y)


plt.show()

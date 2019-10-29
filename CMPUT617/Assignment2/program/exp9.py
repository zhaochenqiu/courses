import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

mean = [0, 0]
cov  = [[1, 0], [0, 1]]
size = 300

x = np.random.multivariate_normal(mean, cov, size)


mean = [0, 0]
cov  = [[1, 0.8], [0.8, 1]]
size = 300

y = np.random.multivariate_normal(mean, cov, size)


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

#    print("joint:",joint.shape)
#    print("marginal:", marginal.shape)

#    print("joint.shape:", joint.shape)
    t = mine_net(joint)
#    print("t.shape:", t.shape)
#    print("t:", t)

#    print("marginal.shape:", marginal.shape)
    et = torch.exp(mine_net(marginal))
#    print("et.shape:", et.shape)

    mi_lb = torch.mean(t) - torch.log(torch.mean(et))

#    print("mi_lb.shape:", mi_lb.shape)
#    print("mi_lb:", mi_lb)

    return mi_lb, t, et

def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         temp1 = data[joint_index, 0]
#         temp2 = data[joint_index][:,0]
#
#         print(temp1)
#         print(temp2)
#
#         subtemp = temp1 - temp2
#         print(subtemp)
#         subvalue = np.sum(np.absolute(subtemp))
#         print(subvalue)

# 两列数据单独采样，会破坏两个随机变量之间的联系
        batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),
                                         data[marginal_index][:,1].reshape(-1,1)],
                                       axis=1)
    return batch


def learn_mine(batch, mine_net, mine_net_optim,  ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch


#     print("joint:",    joint.shape)
#     print("marginal:", marginal.shape)


#    print("input joint:", joint)
    joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
#    print("output joint:", joint)

    marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()


    # mi_lb 是展示的曲线
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)

    # et 是 marginal 的网络输入，和输出
    # ma_et 是迭代变量
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
#    print("ma_et:", ma_et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et))
#    print(loss)
    # use biased estimator
#     loss = - mi_lb

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


# print("y:", len(y))
# print("y type:", type(y))
# print("y.shape:", y.shape)
# print(y)
#
# plt.figure()
# sns.scatterplot(y[:, 0],y[:, 1], color='green')
#
#
# plt.figure()
# joint_data = sample_batch(y,batch_size=100,sample_mode='joint')
# sns.scatterplot(x=joint_data[:,0],y=joint_data[:,1],color='red')
#
# plt.figure()
# marginal_data = sample_batch(y,batch_size=100,sample_mode='marginal')
# sns.scatterplot(x=marginal_data[:,0],y=marginal_data[:,1])
#
# plt.show()

def train(data, mine_net,mine_net_optim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    # data is x or y
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data,batch_size=batch_size), sample_batch(data,batch_size=batch_size,sample_mode='marginal')

#         print(len(batch))
#         print(batch[0].shape)

        # mine_net 输入的网络
        # mine_net_optim 优化器

#         print(len(batch))
#         print(batch[0].shape, batch[1].shape)

        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())

        if (i+1)%(log_freq)==0:
            print(result[-1])
    return result


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]



# mine_net_indep = Mine().cuda()
# # print("11111")
# # print(type(mine_net_indep))
# # print(mine_net_indep)
# # print("22222")
# mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-3)
# result_indep = train(x,mine_net_indep,mine_net_optim_indep)
#
# result_indep_ma = ma(result_indep)
# print(result_indep_ma[-1])
#
# plt.figure()
# # print("len:", len(result_indep_ma))
# plt.plot(range(len(result_indep_ma)),result_indep_ma)



# mine_net_cor = Mine().cuda()
# mine_net_optim_cor = optim.Adam(mine_net_cor.parameters(), lr=1e-3)
# result_cor = train(y,mine_net_cor,mine_net_optim_cor)
#
#
# # print(result_cor.transpose())
# # print(np.transpose(result_cor))
#
# result_cor_ma = ma(result_cor)
# print(result_cor_ma[-1])



# plt.figure()
# plt.plot(range(len(result_cor_ma)),result_cor_ma)
# plt.show()

correlations = np.linspace(-0.9, 0.9, 19)
print(correlations)


plt.figure()

final_result = []
for rho in correlations:
    rho_data = np.random.multivariate_normal( mean=[0,0],
                                  cov=[[1,rho],[rho,1]],
                                 size = 300)

    print("rho_data:", rho_data.shape)
    mine_net = Mine().cuda()
    mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)
    result = train(rho_data,mine_net,mine_net_optim)
    result_ma = ma(result)
    final_result.append(result_ma[-1])
    print(str(rho) + ' : ' + str(final_result[-1]))
    plt.plot(range(len(result_ma)),result_ma)
    plt.pause(0.01)

plt.show()

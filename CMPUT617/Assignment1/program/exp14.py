from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as Data

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import random as rd




def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "\\" + file)

    return re_fs, re_fullfs

def img2patches(img, num = 1000, radius = 100):
    re_data = []

    row, column, byte = img.shape

    for i in range(num):
        r = rd.randint(0, row - 1 - radius)
        c = rd.randint(0, column - 1 - radius)

        patches = img[r:r + radius,c:c + radius,:]

        re_data.append(patches)

    re_data = np.asarray(re_data)
    re_data = re_data.transpose(1, 2, 3, 0)

    return re_data

def linkPatches(patches1, patches2, labs1, labs2):
    patches = np.append(patches1, patches2, axis = 3)
    labs = np.append(labs1, labs2)

    row, column, byte, frames = patches.shape

    index = np.asarray(range(1, frames))

    np.random.shuffle(index)
    np.random.shuffle(index)
    np.random.shuffle(index)


    patches = patches[:, :, :, index]
    labs = labs[index]

    return patches, labs












class MicDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]

        return img, target

    def __len__(self):
        return len(self.images)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

#        print(data.shape)
        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.int64)

#        print(data.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    img_pos = imageio.imread('../imgs/in000001_pos.tif')
    img_neg = imageio.imread('../imgs/in000009_neg.tif')

    num = 10000
    patches_pos = img2patches(img_pos, num, 28)
    patches_neg = img2patches(img_neg, num, 28)

    patches_pos = patches_pos.transpose(3, 2, 0, 1)
    patches_neg = patches_neg.transpose(3, 2, 0, 1)

    patches_pos = patches_pos.dot(1/255)
    patches_neg = patches_neg.dot(1/255)

    labs_pos = np.zeros(num) + 1
    labs_neg = np.zeros(num)

    traindata = np.append(patches_pos, patches_neg, axis=0)
    trainlabs = np.append(labs_pos, labs_neg)

    mytraindata = MicDataset(traindata, trainlabs)



    testnum = 2000
    patches_pos = img2patches(img_pos, testnum, 28)
    patches_neg = img2patches(img_neg, testnum, 28)

    patches_pos = patches_pos.transpose(3, 2, 0, 1)
    patches_neg = patches_neg.transpose(3, 2, 0, 1)

    patches_pos = patches_pos.dot(1/255)
    patches_neg = patches_neg.dot(1/255)


    labs_pos = np.zeros(testnum) + 1
    labs_neg = np.zeros(testnum)

    testdata = np.append(patches_pos, patches_neg, axis = 0)
    testlabs = np.append(labs_pos, labs_neg)

    mytestdata = MicDataset(testdata, testlabs)



    train_loader = torch.utils.data.DataLoader(mytraindata,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(mytestdata,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)



    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torchvision import datasets, transforms
import torch.utils.data

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import random as rd
from PIL import Image

#from torchsummary import summary

def loadFiles_plus(path_im, keyword = ""):
    re_fs = []
    re_fullfs = []

    files = os.listdir(path_im)
    files = sorted(files)

    for file in files:
        if file.find(keyword) != -1:
            re_fs.append(file)
            re_fullfs.append(path_im + "/" + file)

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


def getMicImdb_entry(fullfile, num = 1000, radius = 28):

    lab = -1;
    if fullfile.find("_pos") != -1:
        lab = 1

    if fullfile.find("_neg") != -1:
        lab = 0

    img = imageio.imread(fullfile)

#     row, column, byte = img.shape
#     img = Image.fromarray(img)
#     img = img.resize((int(column/6), int(row/6)))
#     img = np.asarray(img)


    patches = img2patches(img, num, radius)
    patches = patches.dot(1/255)

    labs = np.zeros(num) + lab

    patches = patches.transpose(3, 2, 0, 1)


    return patches, labs


def getMicImdbbyFiles(fullfs, num=1000, radius = 28):


    data_patches, data_labs = getMicImdb_entry(fullfs[-1], num, radius)
    fullfs = fullfs[0:-1]

    for file in fullfs:
        patches, labs = getMicImdb_entry(file, num, radius)

        data_patches = np.append(data_patches, patches, axis = 0)
        data_labs    = np.append(data_labs   , labs   , axis = 0)

    return data_patches, data_labs



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
        self.fc2 = nn.Linear(500, 2)

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



def faketest(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.int64)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probabilityA

            correct += pred.eq(target.view_as(pred)).sum().item()


    pred = np.asarray(pred)

    probability = pred.sum()

#     print(len(pred))
#     print(probability/len(pred))
#     print( 100. * correct / len(test_loader.dataset))

    lab = 0

    if (probability / len(pred)) > 0.5:
        lab = 1
    else:
        lab = 0

    return probability, lab
#
#
#
#
#     value = pred.sum()
#
#
# #     print(pred)
# #
# #     print(target)
# #
# #     print(target.shape)
# #     print(len(target))
#
# #     print(target.view_as(pred))
# #
# #     print(pred)
#
#     print(pred)
#
#     print(pred.dtype)
#
#     print(value)
#
#     print( 100. * correct / len(test_loader.dataset))



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


def evaluation_entry(prolabs, trulabs):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = len(prolabs)

    for i in range(count):
        prolab = int(prolabs[i])
        trulab = int(trulabs[i])

        print("prolab:", prolab, "trulab:", trulab)

        if prolab == 1 and trulab == 1:
            TP += 1
            print("TP")

        if prolab == 1 and trulab == 0:
            FP += 1
            print("FP")

        if prolab == 0 and trulab == 0:
            TN += 1
            print("TN")

        if prolab == 0 and trulab == 1:
            FN += 1
            print("FN")

    print("----------------")
    return TP, TN, FP, FN


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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



    path_im = "/home/cqzhao/dataset/microscopy/src"
#    path_im = "D:\dataset\microscopy"

    fs_pos, fullfs_pos = loadFiles_plus(path_im, "pos")
    fs_neg, fullfs_neg = loadFiles_plus(path_im, "neg")


#     rd.shuffle(fullfs_pos)
#     rd.shuffle(fullfs_neg)

    num_pos = len(fullfs_pos)
    num_neg = len(fullfs_neg)

    num_train_pos = int(np.round(num_pos*0.7))
    num_train_neg = int(np.round(num_neg*0.7))

    num_valid_pos = int(np.round(num_pos*0.2))
    num_valid_neg = int(np.round(num_neg*0.2))

    num_test_pos = int(np.round(num_pos*0.1))
    num_test_neg = int(np.round(num_neg*0.1))


    train_fs = []
    train_fs = np.append(train_fs, fullfs_pos[0:num_train_pos], axis = 0)
    train_fs = np.append(train_fs, fullfs_neg[0:num_train_neg], axis = 0)

    valid_fs = []
    valid_fs = np.append(valid_fs, fullfs_pos[num_train_pos:num_train_pos + num_valid_pos], axis = 0)
    valid_fs = np.append(valid_fs, fullfs_neg[num_train_neg:num_train_neg + num_valid_neg], axis = 0)

    test_fs = []
    test_fs = np.append(test_fs, fullfs_pos[num_train_pos + num_valid_pos:num_train_pos + num_valid_pos + num_test_pos], axis = 0)
    test_fs = np.append(test_fs, fullfs_neg[num_train_neg + num_valid_neg:num_train_neg + num_valid_neg + num_test_neg], axis = 0)


    rd.shuffle(train_fs)
    rd.shuffle(valid_fs)
    rd.shuffle(test_fs)



#     patches_train, labels_train = getMicImdbbyFiles(train_fs)
#     patches_valid, labels_valid = getMicImdbbyFiles(valid_fs)
#
#     index = list(range(len(labels_train)))
#     rd.shuffle(index)
#
#     patches_train = patches_train[index, :, :, :]
#     labels_train = labels_train[index]
#
#
#     index = list(range(len(labels_valid)))
#     rd.shuffle(index)
#
#     patches_valid = patches_valid[index, :, :, :]
#     labels_valid = labels_valid[index]


#     for i in range(100):
#         im = patches_train[i, :, :, :]
#         lb = labels_train[i]
#
#         print(lb)
#
#         im = im.transpose(1, 2, 0)
#
#         plt.imshow(im)
#
#         if lb == 1:
#             plt.title("positive")
#
#         if lb == 0:
#             plt.title("negative")
#
#
#         plt.pause(0.001)

#    print(test_fs)


    list_pro = []
    list_tru = []


    for file in test_fs:
        patches_test, labels_test = getMicImdb_entry(file)

        trulab = 0

        if file.find("pos") != -1:
            trulab = 1
            print("Ground: Positive")

        if file.find("neg") != -1:
            print("Ground: Negative")


        list_tru = np.append(list_tru, trulab)
        testdata  = MicDataset(patches_test,  labels_test)


        test_loader = torch.utils.data.DataLoader(testdata,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


        smodel = model = Net().to(device)
        smodel.load_state_dict(torch.load("mnist_cnn.pt"))


        probability, prolab = faketest(args, model, device, test_loader)

        list_pro = np.append(list_pro, prolab)

        if prolab == 1:
            print('positive')
        else:
            print('negative')

        print(file)

    print(list_pro)
    print(list_tru)


    TP, TN, FP, FN = evaluation_entry(list_pro, list_tru)

    Re = TP/(TP + FN)
    Pr = TP/(TP + FP)

    Fm = (2*Pr*Re)/(Pr + Re)

    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    print("Re:", Re, "Pr:", Pr, "Fm:", Fm)





if __name__ == '__main__':
    main()

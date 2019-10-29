import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
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




# path_im = "/home/cqzhao/dataset/microscopy"
path_im = "D:\dataset\microscopy"

fs_pos, fullfs_pos = loadFiles_plus(path_im, "pos")
fs_neg, fullfs_neg = loadFiles_plus(path_im, "neg")

rd.shuffle(fs_pos)
rd.shuffle(fs_neg)

num_pos = len(fs_pos)
num_neg = len(fs_neg)


num_pos_train = round(num_pos * 0.7)
num_pos_valid = round(num_pos * 0.2)
num_pos_test = round(num_pos * 0.1)


num_neg_train = round(num_neg * 0.7)
num_neg_valid = round(num_neg * 0.2)
num_neg_test = round(num_neg * 0.1)


# print(num_pos_train, num_pos_valid, num_pos_test)
# print(num_neg_train, num_neg_valid, num_neg_test)


fullfs_pos_train = fullfs_pos[0:num_pos_train]
fullfs_neg_train = fullfs_neg[0:num_neg_train]


fullfs_pos_test = fullfs_pos[num_pos - 1 - num_pos_test:num_pos - 1]
fullfs_neg_test = fullfs_neg[num_neg - 1 - num_neg_test:num_neg - 1]


trainim_pos = misc.imread(fullfs_pos_train[0])
trainim_neg = misc.imread(fullfs_neg_train[0])





# data = []
#
# data.append(misc.imread(fullfs_pos[0]))
# data.append(misc.imread(fullfs_pos[1]))
#
#
# im = data[:][:][:][0]
#
# print(im.shape)
#
# plt.imshow(im)
# plt.show()


# data = np.asmatrix(data)

# data = np.mat(data)

# print(data[:][:][:][1])
#
# im = data[:][:][:][0]
#
#
# plt.imshow(im)
# plt.show()
#

#
# for entry in fullfs_pos:
#     im = misc.imread(entry)
#     print(entry)
#     plt.imshow(im)
#     plt.pause(0.01)
#     data.append(im)
#
#
#
#
# im = misc.imread(fullfs_pos[1])
#
# plt.imshow(im)
# plt.show()
# print(im[1, 1, 1])
#


# list_pos = loadFiles_keyword(path_im, "pos")
# list_net = loadFiles_keyword(path_im, "neg")
#
#
# value1, value2 = func_test()
# print(value1)
# print(value2)



# print(list_pos)
# print(list_net)
#
#
# fullpath = path_im + "/" + list_pos[1]
# print(fullpath)
#
#
#
#
# img = mpimg.imread(fullpath)
#
# plt.figure()
# plt.imshow(img)
#
#
# plt.figure()
# plt.imshow(img)
#
# plt.show()


# plt.figure()
# plt.imshow(img)
#
# plt.show()
#
# fullpath1 = path_im + "/" + list_pos[2]
#
# img2 = misc.imread(fullpath1)
# plt.figure()
# plt.imshow(img2)
#
# plt.show()
# # plt.pause(1)





# I = mpimg.imread(path_im + list_pos[1])

#
# files = os.listdir(path_im)
#
# # print(files)
#
#
# list_pos = []
# list_net = []
#
# for file in files:
#     if file.find("pos") != -1:
#         print(file)
#
#
# print(file)
# print(file.find("pos"))
# print(file.find("neg"))
# print(file.find(""))
# # file = files[1]
# #
# # print(file)
# #
# # str = "pos"
# #
# #
# # judge = False
# #
# # print(len(str))
# #
# # for i in range(len(file)):
# #     print(i)

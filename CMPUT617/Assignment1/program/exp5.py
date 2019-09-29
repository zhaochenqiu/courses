import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy import misc
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





# path_im = "/home/cqzhao/dataset/microscopy"
path_im = "D:\dataset\microscopy"

fs_pos, fullfs_pos = loadFiles_plus(path_im, "pos")
fs_neg, fullfs_neg = loadFiles_plus(path_im, "neg")

rd.shuffle(fs_pos)
rd.shuffle(fs_neg)


img_pos = imageio.imread(fullfs_pos[0])
img_neg = imageio.imread(fullfs_neg[0])

# img_pos = imageio.imread('1.png')
# img_neg = imageio.imread('2.png')

num = 100

patches_pos = img2patches(img_pos, num)
patches_neg = img2patches(img_neg, num)

labs_pos = np.zeros(100) + 1
labs_neg = np.zeros(100)


patches, labs = linkPatches(patches_pos, patches_neg, labs_pos, labs_neg)


row, column, byte, frames = patches.shape

for i in range(frames):
    patch = patches[:, :, :, i]

    plt.imshow(patch)
    print("number = %i", i, "label = ", labs[i])
    plt.pause(1)





# print(patches.shape)


# row, column, byte, frames = patches.shape
#
# for i in range(frames):
#     patch = patches[:, :, :, i]
#
#     plt.imshow(patch)
#     plt.pause(0.001)

# print(patches.shape)

# print(patches)
# print(patches.len)
# print(patches.size)

# row, column, byte = img.shape
#
# print(row, column, byte)
#
#
# x = np.random.rand(100)*row
# y = np.random.rand(100)*column
#
# print(x)
# print(y)
#
#
# radius = 100
#
# num = 1000
#
# for i in range(num):
#     r = rd.randint(0, row - 1 - radius)
#     c = rd.randint(0, column - 1 - radius)
#
#     patches = img[r:r + radius,c:c + radius,:]
#
#     print(patches.shape)
#
#     plt.imshow(patches)
#     plt.pause(0.001)

#    print(patches)
#     print(r, c)
#
#     print(len(patches))
#
#     print(patches)

# print('test')
# print(img[55:55+50][665:665+50][:])
# print(img[55][665][:])
#
# print(img[1:3][1:3][:].shape)
#
# print(img[55:55+50, 665:665+50, ...])
#
# print(img[55:55+50, 665:665+50, ...].shape)
# print(img[55:55+50, 665:665+50, :].shape)
#
# print(img[55:55+50, 665:665+50, :].shape)
#



# print(img)
# print(img.shape)
# plt.imshow(img)
# plt.show()



#
# num_pos = len(fs_pos)
# num_neg = len(fs_neg)
#
#
# num_pos_train = round(num_pos * 0.7)
# num_pos_valid = round(num_pos * 0.2)
# num_pos_test = round(num_pos * 0.1)
#
#
# num_neg_train = round(num_neg * 0.7)
# num_neg_valid = round(num_neg * 0.2)
# num_neg_test = round(num_neg * 0.1)
#
#
# # print(num_pos_train, num_pos_valid, num_pos_test)
# # print(num_neg_train, num_neg_valid, num_neg_test)
#
#
# fullfs_pos_train = fullfs_pos[0:num_pos_train]
# fullfs_neg_train = fullfs_neg[0:num_neg_train]
#
#
# fullfs_pos_test = fullfs_pos[num_pos - 1 - num_pos_test:num_pos - 1]
# fullfs_neg_test = fullfs_neg[num_neg - 1 - num_neg_test:num_neg - 1]
#
#
# trainim_pos = misc.imread(fullfs_pos_train[0])
# trainim_neg = misc.imread(fullfs_neg_train[0])
#
#



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

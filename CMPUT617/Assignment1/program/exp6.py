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

    index = np.asarray(range(frames))

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

print(row, column, byte, frames)



for i in range(frames):
    patch = patches[:, :, :, i]

    plt.imshow(patch)
    print("number = %i", i, "label = ", labs[i])
    plt.pause(2)






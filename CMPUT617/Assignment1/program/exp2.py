import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc

def loadFiles_keyword(path_im, keyword):
    files = os.listdir(path_im)

    re_files = []

    for file in files:
        if file.find(keyword) != -1:
            re_files.append(file)

    return re_files


def func_test():
    value = 3;
    value2 = 4;

    return value, value2



path_im = "/home/cqzhao/dataset/microscopy"


list_pos = loadFiles_keyword(path_im, "pos")
list_net = loadFiles_keyword(path_im, "neg")


value1, value2 = func_test()
print(value1)
print(value2)



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

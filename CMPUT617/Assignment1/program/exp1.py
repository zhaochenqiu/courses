import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def loadFiles_keyword(path_im, keyword):
    files = os.listdir(path_im)

    re_files = []

    for file in files:
        if file.find(keyword) != -1:
            re_files.append(file)

    return re_files



path_im = "/home/cqzhao/dataset/microscopy"


list_pos = loadFiles_keyword(path_im, "pos")
list_net = loadFiles_keyword(path_im, "neg")
print(list_pos)
print(list_net)


fullpath = path_im + "/" + list_pos[1]
print(fullpath)

img = mpimg.imread(fullpath)

plt.imshow(img)
plt.show()
# plt.pause(1)




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

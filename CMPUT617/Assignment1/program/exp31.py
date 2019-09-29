import numpy as np
import matplotlib.pyplot as plt
import random as rd

import skimage
from skimage import data
from skimage.transform import pyramid_gaussian


def img2patches(img, num = 1000, radius = 28):
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

def img2patches_pyramid(img, num = 1000, radius = 28):

    pyramid = tuple(pyramid_gaussian(img, downscale=2, multichannel=True))

    img0 = pyramid[0]
    img1 = pyramid[1]
    img2 = pyramid[2]

    row, column, byte = img2.shape

    re_data = []

    for i in range(num):

        r2 = rd.randint(0, row - 1 - radius)
        c2 = rd.randint(0, column - 1 - radius)

        r1 = r2 * 2
        c1 = c2 * 2

        r0 = r1 * 2
        c0 = c1 * 2

        patches0 = img0[r0:r0 + radius,c0:c0 + radius,:]
        patches1 = img1[r1:r1 + radius,c1:c1 + radius,:]
        patches2 = img2[r2:r2 + radius,c2:c2 + radius,:]


        patches = np.append(patches0, patches1, axis = 2)
        patches = np.append(patches, patches2, axis = 2)

        re_data.append(patches)

    re_data = np.asarray(re_data)
    re_data = re_data.transpose(1, 2, 3, 0)

    return re_data






def main():
    # image = data.astronaut()
    img = skimage.io.imread('../imgs/in000001_pos.tif')

    patches = img2patches_pyramid(img)



    print(patches.shape)




#
#     for i in range(10):
#         img2patches_pyramid(img)
#         plt.pause(0.1)
#
#     plt.show()
#
#
#     rows, cols, dim = image.shape
#     pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))
#
#
#
#     composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
#
#     composite_image[:rows, :cols, :] = pyramid[0]
#
#
#     for p in pyramid:
#         print(p.shape)
#     #
#     # i_row = 0
#     # for p in pyramid[1:]:
#     #     n_rows, n_cols = p.shape[:2]
#     #
#     #     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#     #     i_row += n_rows
#     #
#     # fig, ax = plt.subplots()
#     # ax.imshow(composite_image)
#     # plt.show()

if __name__ == '__main__':
    main()

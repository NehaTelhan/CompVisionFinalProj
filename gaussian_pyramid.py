import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import pyramid_gaussian
import skimage.io
import PIL.Image
import glob
import cv2


x_width = 20
y_height = 10
#
filelist = ['AmazonDriveDownload/txt_136.jpg']
#
# for filename in glob.glob('AmazonDriveDownload/*.jpg'):
#     image = skimage.io.imread(filename)
#     # print(image.shape, "Look here")
#
#     rows, cols = image.shape
#     # print(rows, cols)
#     pyramid = tuple(pyramid_gaussian(image, downscale=2))
#
#     composite_image = np.zeros((rows, cols + cols // 2), dtype=np.double)
#
#     composite_image[:rows, :cols] = pyramid[0]
#
#     i_row = 0
#     for p in pyramid[1:]:
#         n_rows, n_cols = p.shape[:2]
#         # print("N_ROWS:", n_rows, "N_COLS:", n_cols)
#         composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
#         i_row += n_rows
#
#     fig, ax = plt.subplots()
#     ax.imshow(composite_image)
#     # print("Done")
#     plt.show()


#for filename in filelist:
for filename in glob.glob('AmazonDriveDownload/*.jpg'):
    image = cv2.imread(filename)
    cv2.imwrite("Original.jpg", image)

    rows = image.shape[0]
    cols = image.shape[1]
    dim = image.shape[2]
    # print("Rows:", rows)
    # print("Cols:", cols)
    # print("Dim:", dim)

    new_rows = int((2*rows)/3)
    new_cols = int((2*cols)/3)
    # print("New Rows:", new_rows)
    # print("New Cols:", new_cols)

    lower_res = cv2.pyrDown(image,dstsize=(16, 16))
    # print(lower_res)
    cv2.imwrite("Lower_res.jpg", lower_res)


    # G = image.copy()
    # gpImage = [G]
    # for i in range(6):
    #     G = cv2.pyrDown(G)
    #     gpImage.append(G)
    #
    # print(gpImage)


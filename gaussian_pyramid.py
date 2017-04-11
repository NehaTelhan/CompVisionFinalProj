import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import pyramid_gaussian, resize
import skimage.io
import PIL.Image
import glob
import cv2


max_x_width = 20
max_y_height = 10
#
filelist = ['AmazonDriveDownload/txt_136.jpg']


''' 
Shows each image in the typical plot scaled down until it can be scaled down no more (1x1)
'''
def gaussian_pyr_one():
    for filename in glob.glob('AmazonDriveDownload/*.jpg'):
        image = skimage.io.imread(filename)
        # print(image.shape, "Look here")

        rows, cols = image.shape
        # print(rows, cols)
        pyramid = tuple(pyramid_gaussian(image, downscale=2))

        composite_image = np.zeros((rows, cols + cols // 2), dtype=np.double)

        composite_image[:rows, :cols] = pyramid[0]

        i_row = 0
        for p in pyramid[1:]:
            n_rows, n_cols = p.shape[:2]
            # print("N_ROWS:", n_rows, "N_COLS:", n_cols)
            composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
            i_row += n_rows

        fig, ax = plt.subplots()
        ax.imshow(composite_image)
        # print("Done")
        plt.show()

''' prints image as an array of values and appends to a super array'''
def gaussian_pyr_two():
    super_arr = []
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
        super_arr.append(lower_res)
        # print("Lower Resolution Array", lower_res)
        # cv2.imwrite("Lower_res.jpg", lower_res)

    print("Super Array", super_arr)

''' prints numerial representation of image as array'''
def gaussain_pyr_three():
    image = 'AmazonDriveDownload/txt_136.jpg'

    G = image.copy()
    gpImage = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpImage.append(G)

    print(gpImage)

''' Prints the image as floats numbers'''
def gaussain_pyr_four():
    image = 'AmazonDriveDownload/txt_136.jpg'
    image = skimage.io.imread(image)

    rows, cols = image.shape
    image = skimage.transform.resize(image, (16, 16))
    plt.show(image)
    print (image)




if __name__ == "__main__":
    # gaussian_pyr_one()
    # gaussian_pyr_two()
    # gaussain_pyr_three()
    gaussain_pyr_four()

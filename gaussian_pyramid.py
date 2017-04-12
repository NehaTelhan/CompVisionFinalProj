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

def gaussian_pyr_one():
    '''
    Shows each image in the typical plot scaled down until it can be scaled down no more (1x1)
    '''
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

def gaussian_pyr_two():
    ''' prints image as an array of values and appends to a super array; also saves the image at it's new lower resolution '''
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
        print("Lower Resolution Array", lower_res)
        cv2.imwrite("Lower_res.jpg", lower_res)

    print("Super Array", super_arr)

def gaussain_pyr_three():
    ''' prints numerial representation of image as array'''
    image = 'AmazonDriveDownload/txt_136.jpg'

    G = image.copy()
    gpImage = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpImage.append(G)

    print(gpImage)

def gaussain_pyr_four():
    ''' Prints the image as floats numbers'''

    image = 'AmazonDriveDownload/txt_136.jpg'
    image = skimage.io.imread(image)

    rows, cols = image.shape
    image = skimage.transform.resize(image, (20, 10))
    plt.show(image)
    print (image)


def gaussian_pyr_five():
    super_arr = []
    for filename in glob.glob('AmazonDriveDownload/*.jpg'):
        image = skimage.io.imread(filename)
        rows = image.shape[0]
        cols = image.shape[1]
        # print("Rows:", rows)
        # print("Cols:", cols)
        # print("Dim:", dim)
        image_arr = skimage.transform.resize(image, (int((2/3)*rows), int((2/3)*cols)))
        # image_arr = skimage.transform.resize(image, (20, 10))
        # print(image_arr)
        super_arr.append(image_arr)
    print(super_arr)



if __name__ == "__main__":
    # gaussian_pyr_one()
    # gaussian_pyr_two()
    # gaussain_pyr_three()
    # gaussain_pyr_four()
    gaussian_pyr_five()

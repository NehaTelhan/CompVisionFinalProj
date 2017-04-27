import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import glob
import cv2

def center_histogram(coordinate, rows, cols):
    '''
    Input is:
        - I NEED TO CALCULATE THE FOLLOWING BASED OFF THE ORIGINAL IMAGE: start coordinate, height of image, width of image
        - the "box" around the image as denoted by a start coordinate, height of that "box", and width of the "box"
        - So take in original picture, the coordinate of the start of the box on the text_box, the number of rows (height) of that textbox and the number of columns of that textbox (width)
    :return:
        Two color histograms?!
    '''

    # for filename in glob.glob('NehaRandoImages/*.jpg'):
    #     image = skimage.io.imread(filename)
    original_image_path = 'NehaRandoImages/file0001625591306.jpg' #Super Autumn Colors
    # original_image_path = 'NehaRandoImages/file0001735386118.jpg' #Super Blue Image
    original_image = cv2.imread(original_image_path)
    color = ('b', 'g', 'r')
    original_rows = original_image.shape[0]
    original_cols = original_image.shape[1]

    box_height = rows
    box_width = cols
    x_pix = coordinate[0]
    y_pix = coordinate[1]

    #An array to hold the four center rows of the textbox
    center_four_rows = []

    '''Finds the center of the textbox
    and adds the two rows above and below the middle line to array
    from the left of the textbox to the right'''
    for i in range((y_pix+box_height//2)-2, (y_pix+box_height//2)+2):
        center_four_rows.append(original_image[i][x_pix:x_pix+box_width])

    #Make the color histogram for the text box's center rows and show the colors (RGB) as lines
    for i, col in enumerate(color):
        histrA = cv2.calcHist(center_four_rows, [i], None, [256], [0, 256])
        # cv2.normalize(histrA, histrA, 0, 255, cv2.NORM_MINMAX) #do I even need this line?
        plt.title('Histogram for Text Boxs 4 Center Rows')
        plt.plot(histrA, color=col)
        plt.xlim([0,256])
    plt.show()
    print("Histogram A: ", histrA)

    '''An array to hold the four outside rows
    two rows from the top of the outside of the textbox
    two rows from the bottom of the outside of the textbox'''
    outer_four_rows = []

    '''Loops through the two entire rows from edge to edge of picture
    two rows above...'''
    for i in range((y_pix - 2), y_pix):
        outer_four_rows.append(original_image[i])

    #two rows below....
    for i in range((y_pix + box_height), y_pix+box_height+2):
        outer_four_rows.append(original_image[i])

    #Make the color histogram for the whole images two upper + two lower rows show the colors (BGR) as lines
    for i, col in enumerate(color):
        histrB = cv2.calcHist(outer_four_rows, [i], None, [256], [0, 256])
        # cv2.normalize(histrB, histrB, 0, 255, cv2.NORM_MINMAX)#do I even need this line?
        plt.title('Histogram for Entire Images 2 Upper/Lower Rows')
        plt.plot(histrB, color=col)
        plt.xlim([0, 256])
    plt.show()
    print("Histogram B: ", histrB)

    '''
    max value difference between two histograms is the intensity of the greatest difference in color (intensity = avg od color)
    (SSD)
    '''
    # print("HistrA", histrA, len(histrA))
    # print("HistrB", histrB, len(histrB))
    ssd = []
    for a in histrA:
        for b in histrB:
            diff = a - b
            # print("A", a, "B", b , "DIFF" , diff)
            ssd.append(diff**2)
    print("MAX", max(ssd))

    print("Compare:", cv2.compareHist(histrA, histrB, 1))

    # print("original_rows", original_rows)
    # print("original cols", original_cols)
    # print("X_coord", x_pix)
    # print("Y_coord", y_pix)


if __name__ == "__main__":
    center_histogram(coordinate=(20,34), rows=40, cols=120)


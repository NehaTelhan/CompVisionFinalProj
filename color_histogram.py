import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import glob
import cv2

def ultimate_ans(max, min):
    if (max < min):
        return False
    else:
        return True

''' Sadiyah's output is a list of lists where the inner list has this general outline ----- [ [start_x, start_y, text_cols, text_rows]... ]'''

def color_histogram(original_image_path, list):
    '''
    Input is: Runs on single image containing single textbox
        - I NEED TO CALCULATE THE FOLLOWING BASED OFF THE ORIGINAL IMAGE: start coordinate(0,0), height of image, width of image
        - the "box" around the text as denoted by a start coordinate, height of that "box", and width of the "box"
    :return:
        true or false based on whether the intensity in that image happens closer to the 0 color or 256 color
    '''
    # original_image_path = 'NehaRandoImages/file0001625591306.jpg' #Super Autumn Colors
    # original_image_path = 'NehaRandoImages/file0001735386118.jpg' #Super Blue Image
    # original_image_path = 'NehaRandoImages/blackcirclewhitebackground.png'
    original_image = cv2.imread(original_image_path)
    color = ('b', 'g', 'r')
    original_rows = original_image.shape[0]
    original_cols = original_image.shape[1]
    # print("Image height:", original_rows)
    # print("Image width:", original_cols)

    box_height = list[3]
    box_width = list[2]
    x_pix = list[0]
    y_pix = list[1]

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
        plt.ylabel("# of pixels")
        plt.xlabel("bins")
    plt.show()

    '''An array to hold the four outside rows
    two rows from the top of the outside of the textbox
    two rows from the bottom of the outside of the textbox'''
    outer_four_rows = []

    '''Loops through the two entire rows from edge to edge of picture
    two rows above...'''
    #Row right above textbox and first row of the textbox
    for i in range((y_pix - 1), (y_pix+1)):
        outer_four_rows.append(original_image[i])

    #one row below textbox and the last row of textbox....
    for i in range((y_pix + box_height-1), y_pix+box_height+1):
        outer_four_rows.append(original_image[i])

    #Make the color histogram for the whole images two upper + two lower rows show the colors (BGR) as lines
    for i, col in enumerate(color):
        histrB = cv2.calcHist(outer_four_rows, [i], None, [256], [0, 256])
        # cv2.normalize(histrB, histrB, 0, 255, cv2.NORM_MINMAX)#do I even need this line?
        plt.title('Histogram for Entire Images 2 Upper/Lower Rows')
        plt.plot(histrB, color=col)
        plt.xlim([0, 256])
        plt.ylabel("# of pixels")
        plt.xlabel("bins")
    plt.show()
    '''
    max value difference between two histograms is the intensity of the greatest difference in color (intensity = avg od color)
    256 slots = intensities number inside slot = number of pixels that are of that intensity
        '''
    # print("HistrA", histrA, len(histrA))
    # print("HistrB", histrB, len(histrB))
    diff_array = []
    for a in histrA:
        for b in histrB:
            diff = abs(a - b)
            diff_array.append(diff)

    # Maxxy is how many pixels have that intensity
    act_min = min(diff_array)
    min_positions = [i for i, x in enumerate(diff_array) if x == act_min]

    act_max = max(diff_array)
    max_positions = [i for i, x in enumerate(diff_array) if x == act_max]

    ans = ultimate_ans(max_positions[0], min_positions[0])
    print(ans)
    return ans


if __name__ == "__main__":

    # image = cv2.imread("NehaRandoImages/textonly.png")
    # print("1", image.shape)
    #
    image2 = cv2.imread("NehaRandoImage/outside.png")
    print("2", image2.shape)
    #
    # color_histogram("NehaRandoImages/textonly.png", [0, 0, image.shape[1]-1, image.shape[0]-1 ])
    color_histogram("NehaRandoImages/outside.png", [0, 0, image2.shape[1]-1, image2.shape[0]-1 ])


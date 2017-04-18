""" Vertical and Horizontal Segmentation methods """
import numpy


def set_up(text_box, image):
    # text_box = [starting_pixel, width, height]
    starting_pixel = text_box[0] # should be a tuple, (x, y)
    width = text_box[1] # num columns
    height = text_box[2] # num rows
    full_box = numpy.empty(width, height)
    offset = int(height/2) # SHOULD I BE CASTING TO INT HERE IDK
    starting_pixel[1] = starting_pixel[1] - offset
    height = height + offset

    # building the full text_box to do the projection on
    for x in range(0, width):
        for y in range(0, height):
            full_box[x][y] = image[starting_pixel[0]+x][starting_pixel[1]+y]

    # vertical projection = sum of pixel intensities over every row
    vert_proj = [sum(full_box[i]) for i in range(height)]
    min_vert = min(vert_proj)
    max_vert = max(vert_proj)

    # horizontal projection = sum of pixel intensities over every column
    horizontal_proj = [sum(full_box[:, i]) for i in range(width)]
    min_horiz = min(horizontal_proj)
    max_horiz = max(horizontal_proj)

    # calculate the segmentation threshold
    # WHAT IS A SEGMENTATION THRESHOLD
    # THESE VALUES WILL CHANGE ITS JUST BC I WANNA SET UP LOGIC OF CODE
    vert_seg_thresh = (min_vert, max_vert)
    horiz_seg_thresh = (min_horiz, max_horiz)
    new_box = [starting_pixel, width, height]
    vertical(vert_seg_thresh, vert_proj, new_box)
    horizontal(horiz_seg_thresh, horizontal_proj, new_box)


def vertical(vert_threshold, vert_proj, box):
    """ loop through all rows of profile to set boundaries """


def horizontal(horiz_threshold, horizontal_proj, box):
    """ loop through all columns of profile to set boundaries """

if __name__ == "__main__":
    # set_up(text_box, image)
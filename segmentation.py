""" Vertical and Horizontal Segmentation methods """
import numpy
from text_extractor import saliency_map


def initial_box():
    print("got to initial_box")
    return saliency_map("demo-image1.jpg")


def set_up(text_box, saliency_map):
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
            full_box[x][y] = saliency_map[starting_pixel[0] + x][starting_pixel[1] + y]

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
    vert_box_list = vertical(vert_seg_thresh, vert_proj, new_box, height)
    horiz_box_list = horizontal(horiz_seg_thresh, horizontal_proj, new_box, width)

    return [vert_box_list, horiz_box_list]


def vertical(vert_threshold, vert_proj, box, height):
    """ loop through all rows of profile to set boundaries """
    change = False
    upper_bound = None
    lower_bound = None
    box_list = []
    for i in range(0, height):
        if vert_proj[i] > vert_threshold[0]:
            if upper_bound is None:
                upper_bound = i
        else:
            if lower_bound is None:
                lower_bound = i
            if upper_bound is not None:
                new_box = [box[0], upper_bound, lower_bound]
                box_list.append(new_box)
                upper_bound = None
                lower_bound = None
                change = True
    return box_list


def horizontal(horiz_threshold, horizontal_proj, box, width):
    """ loop through all columns of profile to set boundaries """
    left_bound = None
    right_bound = None
    box_list = []
    for i in range(0, width):
        if horizontal_proj[i] > horiz_threshold[0]:
            if left_bound is None:
                left_bound = i
            elif right_bound is not None:
                if abs(i-right_bound) > horiz_threshold[0]: # wth is large enough?
                    new_box = [box[0], left_bound, right_bound]
                    box_list.append(new_box)
                    left_bound = None
                    right_bound = None
                else:
                    right_bound = None

        elif right_bound is None:
            right_bound = i
    if left_bound is not None and right_bound is None:
        right_bound = width
    if left_bound is not None and right_bound is not None:
        new_box = [box[0], left_bound, right_bound]
        box_list.append(new_box)

    return box_list


if __name__ == "__main__":
    ret = initial_box()
    print("boxes", ret[0])
    print("sal_map", ret[1])
    # print("Real saliency")
    # print(ret[0])
    # all_boxes = []
    # for box in ret[0]:
    #     all_boxes.append(set_up(box, ret[1]))
    # print("Printing refined boxes: ", all_boxes)

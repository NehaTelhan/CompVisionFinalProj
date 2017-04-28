""" Vertical and Horizontal Segmentation methods """
import numpy
from text_extractor import saliency_map
BOX_LIST = []


def initial_box():
    print("got to initial_box")
    return saliency_map("demo-image1.jpg")


def set_up(text_box, saliency_map):
    # text_box = [starting_pixel, width, height]
    starting_pixel = [text_box[0], text_box[1]] # should be a tuple, (x, y)
    width = int(text_box[2]) # num columns
    height = int(text_box[3]) # num rows
    offset = int(height/2) # SHOULD I BE CASTING TO INT HERE IDK
    starting_pixel[1] = starting_pixel[1] - offset
    height = height + offset

    # building the full text_box to do the projection on
    # print("width", width)
    # print("height", height)
    if height > saliency_map.shape[0]:
        height = saliency_map.shape[0]
    if width > saliency_map.shape[1]:
        width = saliency_map.shape[1]
    full_box = numpy.empty((height, width))
    for x in range(0, height):
        for y in range(0, width):
            i = starting_pixel[0] + x
            j = starting_pixel[1] + y
            # print("i", i)
            # print("j", j)
            if i >= width and j >= height:
                full_box[x][y] = saliency_map[width][height]
                continue
            elif i >= width and j < height:
                full_box[x][y] = saliency_map[width][j]
                continue
            elif i < width and j >= height:
                full_box[x][y] = saliency_map[i][height]
                continue
            else:
                # print(height, width)
                # print(full_box.shape[0], full_box.shape[1])
                # print("sal_map", saliency_map.shape[0], saliency_map.shape[1])
                # print("x", x, "y", y)
                # print("i", i, "j", j)
                full_box[x][y] = saliency_map[i][j]
                continue

    print("trying to do vert projections")
    # vertical projection = sum of pixel intensities over every row
    # vert_proj = [sum(full_box[i]) for i in range(height)]
    vert_proj = numpy.sum(full_box, axis=0)
    min_vert = min(vert_proj)
    max_vert = max(vert_proj)

    print("trying to do horizontal projections")
    # horizontal projection = sum of pixel intensities over every column
    horizontal_proj = numpy.sum(full_box, axis=1)
    # horizontal_proj = [sum(full_box[i]) for i in range(width)]
    min_horiz = min(horizontal_proj)
    max_horiz = max(horizontal_proj)

    # calculate the segmentation threshold
    # WHAT IS A SEGMENTATION THRESHOLD
    # THESE VALUES WILL CHANGE ITS JUST BC I WANNA SET UP LOGIC OF CODE
    vert_seg_thresh = (min_vert, max_vert)
    horiz_seg_thresh = (min_horiz, max_horiz)
    new_box = [starting_pixel[0], starting_pixel[1], width, height]
    vertical(vert_seg_thresh, vert_proj, new_box, height)
    horizontal(horiz_seg_thresh, horizontal_proj, new_box, width)


def vertical(vert_threshold, vert_proj, box, height):
    """ loop through all rows of profile to set boundaries """
    change = False
    upper_bound = None
    lower_bound = None
    for i in range(0, len(vert_proj)):
        # print("i", i, "height", height)
        if vert_proj[i] > vert_threshold[0]:
            if upper_bound is None:
                upper_bound = i
        else:
            if lower_bound is None:
                lower_bound = i
            if upper_bound is not None:
                # new_box = [box[0], box[1], upper_bound, lower_bound]
                BOX_LIST.append([box[0], box[1], upper_bound, lower_bound])
                upper_bound = None
                lower_bound = None
                change = True


def horizontal(horiz_threshold, horizontal_proj, box, width):
    """ loop through all columns of profile to set boundaries """
    left_bound = None
    right_bound = None
    for i in range(0, len(horizontal_proj)):
        if horizontal_proj[i] > horiz_threshold[0]:
            if left_bound is None:
                left_bound = i
            elif right_bound is not None:
                if abs(i-right_bound) > horiz_threshold[0]: # wth is large enough?
                    # new_box = [box[0], box[1], left_bound, right_bound]
                    BOX_LIST.append([box[0], box[1], left_bound, right_bound])
                    left_bound = None
                    right_bound = None
                else:
                    right_bound = None

        elif right_bound is None:
            right_bound = i
    if left_bound is not None and right_bound is None:
        right_bound = width
    if left_bound is not None and right_bound is not None:
        # new_box = [box[0], box[1], left_bound, right_bound]
        BOX_LIST.append([box[0], box[1], left_bound, right_bound])


if __name__ == "__main__":
    ret = initial_box()
    # print("boxes", ret[0])
    # print("sal_map", ret[1])
    # print("Real saliency")
    # print(ret[0])
    for box in ret[0]:
        set_up(box, ret[1])
    # print(BOX_LIST)
    print("length of originals", len(ret[0]))
    print("len Box List", len(BOX_LIST))
    outfile = open("seg_boxes.txt", 'w')
    for item in BOX_LIST:
        outfile.write("%s\n" % item)
    print(ret[0])
    print("end")
    # old_boxes = ret[0]
    # same = []
    # offset = 100
    # for i in range(0, len(old_boxes)):
    #     old_x = old_boxes[i][0]
    #     old_y = old_boxes[i][1]
    #     for j in range(0, len(BOX_LIST)):
    #         # print(BOX_LIST[j])
    #         vert_x = BOX_LIST[j][0]
    #         vert_y = BOX_LIST[j][1]
    #         if vert_x + offset > old_x > vert_x - offset and vert_y + offset > old_y > vert_y - offset:
    #             same.append(BOX_LIST[j])
    # print("len same", len(same))
    # print(same)

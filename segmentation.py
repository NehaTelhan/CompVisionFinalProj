""" Vertical and Horizontal Segmentation methods """
import numpy
#from text_extractor import saliency_map
BOX_LIST = []
FINAL_BOXES = []


def initial_box():
    print("got to initial_box")
    #return saliency_map("demo-image1.jpg")

def projection_set_up(starting_pixel, height, width, saliency_map):
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
    return full_box

def set_up(text_box, saliency_map):
    # text_box = [starting_pixel, width, height]
    starting_pixel = [text_box[0], text_box[1]] # should be a tuple, (x, y)
    width = int(text_box[2]) # num columns
    height = int(text_box[3]) # num rows
    #offset = int(height/2) # SHOULD I BE CASTING TO INT HERE IDK
    # offset = 10
    # starting_pixel[1] = starting_pixel[1] - int(offset/2)
    # starting_pixel[0] = starting_pixel[0] - int(offset/2)
    # height = height + offset
    # width = width + offset

    # building the full text_box to do the projection on
    # print("width", width)
    # print("height", height)
    if height > saliency_map.shape[0]:
        height = saliency_map.shape[0]
    if width > saliency_map.shape[1]:
        width = saliency_map.shape[1]
    print(height, width, text_box[3], text_box[2])
    full_box = projection_set_up(starting_pixel, height, width, saliency_map)

    print("trying to do vert projections")
    # vertical projection = sum of pixel intensities over every row
    # vert_proj = [sum(full_box[i]) for i in range(height)]
    vert_proj = numpy.sum(full_box, axis=1)
    min_vert = min(vert_proj)
    max_vert = max(vert_proj)
    print("Proj:", vert_proj)

    vert_seg_thresh = min_vert
    new_box = [starting_pixel[0], starting_pixel[1], width, height]
    boxes = vertical(vert_seg_thresh, vert_proj, new_box)
    print("boxes", boxes)

    print("trying to do horizontal projections")
    # horizontal projection = sum of pixel intensities over every column

    # horizontal_proj = [sum(full_box[i]) for i in range(width)]


    # calculate the segmentation threshold
    # WHAT IS A SEGMENTATION THRESHOLD
    # THESE VALUES WILL CHANGE ITS JUST BC I WANNA SET UP LOGIC OF CODE
    #vert_seg_thresh = (min_vert, max_vert)
    #horiz_seg_thresh = (min_horiz, max_horiz)
    for box in boxes:
        full_box = projection_set_up([box[0], box[1]], box[3], box[2], saliency_map)
        #horizontal_proj = numpy.sum(full_box, axis=0)
        horizontal_proj = numpy.zeros(box[2])
        for j in range(box[2]):
            sum = 0
            for i in range(box[3]):
                sum += saliency_map[i, j]
            horizontal_proj[j] = sum

        print(horizontal_proj)
        min_horiz = min(horizontal_proj)
        max_horiz = max(horizontal_proj)
        print("len", len(horizontal_proj))
        horiz_seg_thresh = min_horiz + int((max_horiz - min_horiz) * 0.025)
        print(horiz_seg_thresh)

        horizontal(horiz_seg_thresh, horizontal_proj, box)

def vertical(vert_threshold, vert_proj, box):
    """ loop through all rows of profile to set boundaries """
    boxes = []
    change = False
    upper_bound = None
    lower_bound = None
    for i in range(0, len(vert_proj)):
        # print("i", i, "height", height)
        if vert_proj[i] > vert_threshold:
            if upper_bound is None:
                upper_bound = i
        else:
            lower_bound = i
            if upper_bound is not None:
                # new_box = [box[0], box[1], upper_bound, lower_bound]
                print(upper_bound, lower_bound)
                if lower_bound - upper_bound + 1 > 25:
                    boxes.append([box[0] + upper_bound, box[1], box[2], lower_bound - upper_bound + 1])
                upper_bound = None
                lower_bound = None
                change = True
    if not upper_bound == None:
        print("upper", len(vert_proj))
        if len(vert_proj) - upper_bound > 25:
            boxes.append([box[0] + upper_bound, box[1], box[2], len(vert_proj) - upper_bound])
    return boxes

def horizontal(horiz_threshold, horizontal_proj, box):
    """ loop through all columns of profile to set boundaries """
    left_bound = None
    right_bound = None
    gap = box[3] / 20.0 # height of current text box
    for i in range(0, len(horizontal_proj)):
        if horizontal_proj[i] > horiz_threshold:
            if left_bound is None:
                left_bound = i
            elif right_bound is not None:
                if abs(i-right_bound) > gap: # wth is large enough?
                    # new_box = [box[0], box[1], left_bound, right_bound]
                    if (right_bound - left_bound + 1 > 25):
                        FINAL_BOXES.append([box[0], box[1] + left_bound, right_bound - left_bound + 1, box[3]])
                    left_bound = None
                    right_bound = None
                else:
                    right_bound = None
        else:
            if right_bound is None:
                right_bound = i

    if left_bound is not None and right_bound is None:
        right_bound = len(horizontal_proj) - 1
        if right_bound - left_bound + 1 > 25:
            FINAL_BOXES.append([box[0], box[1] + left_bound, right_bound - left_bound + 1, box[3]])
    elif left_bound is not None and right_bound is not None:
        # new_box = [box[0], box[1], left_bound, right_bound]
        if right_bound - left_bound + 1 > 25:
            FINAL_BOXES.append([box[0], box[1] + left_bound, right_bound - left_bound + 1, box[3]])

def run_segmentation(initial_boxes, saliency_map):
    for box in initial_boxes:
        set_up(box, saliency_map)
    return FINAL_BOXES

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

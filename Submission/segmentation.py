""" Vertical and Horizontal Segmentation methods """
import numpy
BOX_LIST = []
FINAL_BOXES = []

def projection_set_up(starting_pixel, height, width, saliency_map):
    full_box = numpy.empty((height, width))
    for x in range(0, height):
        for y in range(0, width):
            i = starting_pixel[0] + x
            j = starting_pixel[1] + y

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
                full_box[x][y] = saliency_map[i][j]
                continue
    return full_box

def set_up(text_box, saliency_map):
    starting_pixel = [text_box[0], text_box[1]] # should be a tuple, (x, y)
    width = int(text_box[2]) # num columns
    height = int(text_box[3]) # num rows

    # building the full text_box to do the projection on
    if height > saliency_map.shape[0]:
        height = saliency_map.shape[0]
    if width > saliency_map.shape[1]:
        width = saliency_map.shape[1]
    print(height, width, text_box[3], text_box[2])
    full_box = projection_set_up(starting_pixel, height, width, saliency_map)

    print("trying to do vert projections")
    # vertical projection = sum of pixel intensities over every row
    vert_proj = numpy.sum(full_box, axis=1)
    min_vert = min(vert_proj)
    print("Proj:", vert_proj)

    vert_seg_thresh = min_vert
    new_box = [starting_pixel[0], starting_pixel[1], width, height]
    boxes = vertical(vert_seg_thresh, vert_proj, new_box)
    print("boxes", boxes)

    print("trying to do horizontal projections")
    for box in boxes:
        horizontal_proj = numpy.zeros(box[2])
        for j in range(box[2]):
            sum = 0
            for i in range(box[3]):
                sum += saliency_map[i + box[0], j + box[1]]
            horizontal_proj[j] = sum

        print(horizontal_proj)
        min_horiz = min(horizontal_proj)
        horiz_seg_thresh = min_horiz

        horizontal(horiz_seg_thresh, horizontal_proj, box)

def vertical(vert_threshold, vert_proj, box):
    """ loop through all rows of profile to set boundaries """
    boxes = []
    upper_bound = None
    lower_bound = None
    for i in range(0, len(vert_proj)):
        if vert_proj[i] > vert_threshold:
            if upper_bound is None:
                upper_bound = i
        else:
            lower_bound = i
            if upper_bound is not None:
                print(upper_bound, lower_bound)
                if lower_bound - upper_bound + 1 > 25:
                    boxes.append([box[0] + upper_bound, box[1], box[2], lower_bound - upper_bound + 1])
                upper_bound = None
                lower_bound = None
    if not upper_bound == None:
        print("upper", len(vert_proj))
        if len(vert_proj) - upper_bound > 25:
            boxes.append([box[0] + upper_bound, box[1], box[2], len(vert_proj) - upper_bound])
    return boxes

def horizontal(horiz_threshold, horizontal_proj, box):
    """ loop through all columns of profile to set boundaries """
    left_bound = None
    right_bound = None
    gap = box[3] / 10.0 # height of current text box
    for i in range(0, len(horizontal_proj)):
        if horizontal_proj[i] > horiz_threshold:
            if left_bound is None:
                left_bound = i
            elif right_bound is not None:
                if abs(i-right_bound) > gap:
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
        if right_bound - left_bound + 1 > 25:
            FINAL_BOXES.append([box[0], box[1] + left_bound, right_bound - left_bound + 1, box[3]])

def run_segmentation(initial_boxes, saliency_map):
    for box in initial_boxes:
        set_up(box, saliency_map)
    return FINAL_BOXES

if __name__ == "__main__":
    pass
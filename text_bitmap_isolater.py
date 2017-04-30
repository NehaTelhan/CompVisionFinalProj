import math
import numpy
import pylab
import skimage.color, skimage.io

# Input original image in color as float array
def remove_background(original_image, box_start_row, box_start_col, box_width, box_height, is_text_inverse):
    original_image = skimage.img_as_float(original_image)
    threshold_seedfill = 1.1

    height = original_image.shape[0]
    width = original_image.shape[1]

    box_end_row = box_start_row + box_height - 1
    box_end_col = box_start_col + box_width - 1

    # Increase text bounding box
    # width_expansion = int(box_width * 0.1)
    # if box_start_col <= width_expansion:
    #     box_start_col = 0
    # else:
    #     box_start_col = box_start_col - width_expansion
    # if box_end_col + width_expansion >= width:
    #     box_end_col = width - 1
    # else:
    #     box_end_col = box_end_col + width_expansion
    #
    # height_expansion = int(box_height * 0.2)
    # if box_start_row <= height_expansion:
    #     box_start_row = 0
    # else:
    #     box_start_row = box_start_row - height_expansion
    # if box_end_row + height_expansion >= height:
    #     box_end_row = height - 1
    # else:
    #     box_end_row = box_end_row + height_expansion

    # Background color
    color = [1, 1, 1]
    if is_text_inverse:
        color = [0, 0, 0]

    # box_height = box_end_row - box_start_row + 1
    # box_width = box_end_col - box_start_col + 1
    box = numpy.zeros((box_height, box_width))

    # Take pixel on boundary as seed to fill all pixels with background color which do not differ more than threshold
    # left
    for i in range(box_height):
        pixels_to_check = [(i, 0)]
        boundary_color = original_image[i + box_start_row, box_start_col]
        box[i, 0] = 1
        while len(pixels_to_check) > 0:
            pixel = pixels_to_check.pop(0)
            if calc_rgb_distance(original_image[pixel[0] + box_start_row, pixel[1] + box_start_col], boundary_color) < threshold_seedfill:
                original_image[pixel[0] + box_start_row, pixel[1] + box_start_col] = color
                # left
                if pixel[1] - 1 >= 0 and box[pixel[0], pixel[1] - 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] - 1))
                    box[pixel[0], pixel[1] - 1] = 1
                # top
                if pixel[0] - 1 >= 0 and box[pixel[0] - 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] - 1, pixel[1]))
                    box[pixel[0] - 1, pixel[1]] = 1
                # right
                if pixel[1] + 1 < box_width and box[pixel[0], pixel[1] + 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] + 1))
                    box[pixel[0], pixel[1] + 1] = 1
                # bottom
                if pixel[0] + 1 < box_height and box[pixel[0] + 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] + 1, pixel[1]))
                    box[pixel[0] + 1, pixel[1]] = 1
    # top
    for j in range(box_width):
        pixels_to_check = [(0, j)]
        boundary_color = original_image[box_start_row, box_start_col + j]
        box[0, j] = 1
        while len(pixels_to_check) > 0:
            pixel = pixels_to_check.pop(0)
            if calc_rgb_distance(original_image[pixel[0] + box_start_row, pixel[1] + box_start_col],
                                 boundary_color) < threshold_seedfill:
                original_image[pixel[0] + box_start_row, pixel[1] + box_start_col] = color
                # left
                if pixel[1] - 1 >= 0 and box[pixel[0], pixel[1] - 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] - 1))
                    box[pixel[0], pixel[1] - 1] = 1
                # top
                if pixel[0] - 1 >= 0 and box[pixel[0] - 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] - 1, pixel[1]))
                    box[pixel[0] - 1, pixel[1]] = 1
                # right
                if pixel[1] + 1 < box_width and box[pixel[0], pixel[1] + 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] + 1))
                    box[pixel[0], pixel[1] + 1] = 1
                # bottom
                if pixel[0] + 1 < box_height and box[pixel[0] + 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] + 1, pixel[1]))
                    box[pixel[0] + 1, pixel[1]] = 1
    # right
    for i in range(box_height):
        pixels_to_check = [(i, box_width - 1)]
        boundary_color = original_image[i + box_start_row, box_start_col + box_width - 1]
        box[i, box_width - 1] = 1
        while len(pixels_to_check) > 0:
            pixel = pixels_to_check.pop(0)
            if calc_rgb_distance(original_image[pixel[0] + box_start_row, pixel[1] + box_start_col],
                                 boundary_color) < threshold_seedfill:
                original_image[pixel[0] + box_start_row, pixel[1] + box_start_col] = color
                # left
                if pixel[1] - 1 >= 0 and box[pixel[0], pixel[1] - 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] - 1))
                    box[pixel[0], pixel[1] - 1] = 1
                # top
                if pixel[0] - 1 >= 0 and box[pixel[0] - 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] - 1, pixel[1]))
                    box[pixel[0] - 1, pixel[1]] = 1
                # right
                if pixel[1] + 1 < box_width and box[pixel[0], pixel[1] + 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] + 1))
                    box[pixel[0], pixel[1] + 1] = 1
                # bottom
                if pixel[0] + 1 < box_height and box[pixel[0] + 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] + 1, pixel[1]))
                    box[pixel[0] + 1, pixel[1]] = 1
    # bottom
    for j in range(box_width):
        pixels_to_check = [(box_height - 1, j)]
        boundary_color = original_image[box_start_row + box_height - 1, box_start_col + j]
        box[box_height - 1, j] = 1
        while len(pixels_to_check) > 0:
            pixel = pixels_to_check.pop(0)
            if calc_rgb_distance(original_image[pixel[0] + box_start_row, pixel[1] + box_start_col],
                                 boundary_color) < threshold_seedfill:
                original_image[pixel[0] + box_start_row, pixel[1] + box_start_col] = color
                # left
                if pixel[1] - 1 >= 0 and box[pixel[0], pixel[1] - 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] - 1))
                    box[pixel[0], pixel[1] - 1] = 1
                # top
                if pixel[0] - 1 >= 0 and box[pixel[0] - 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] - 1, pixel[1]))
                    box[pixel[0] - 1, pixel[1]] = 1
                # right
                if pixel[1] + 1 < box_width and box[pixel[0], pixel[1] + 1] == 0:
                    pixels_to_check.append((pixel[0], pixel[1] + 1))
                    box[pixel[0], pixel[1] + 1] = 1
                # bottom
                if pixel[0] + 1 < box_height and box[pixel[0] + 1, pixel[1]] == 0:
                    pixels_to_check.append((pixel[0] + 1, pixel[1]))
                    box[pixel[0] + 1, pixel[1]] = 1

    # potentially add algorithm for eight-connect to delete more regions
    return binarize_bitmap(original_image, box_start_row, box_start_col, box_width, box_height, is_text_inverse)

def calc_rgb_distance(color1, color2):
    return math.sqrt((color1[0] - color2[0]) * (color1[0] - color2[0]) + (color1[1] - color2[1]) * (color1[1] - color2[1]) + (color1[2] - color2[2]) * (color1[2] - color2[2]))

def binarize_bitmap(original_image, box_start_row, box_start_col, box_width, box_height, is_text_inverse):
    grayscale = skimage.color.rgb2gray(original_image)
    text_box_bitmap = numpy.zeros((box_height, box_width))

    binarization_threshold = 0.5
    for y in range(box_height):
        for x in range(box_width):
            if is_text_inverse:
                # text is white
                if grayscale[y + box_start_row, x + box_start_col] < binarization_threshold:
                    text_box_bitmap[y, x] = 1
                else:
                    text_box_bitmap[y, x] = 0
            else:
                if grayscale[y + box_start_row, x + box_start_col] > binarization_threshold:
                    text_box_bitmap[y, x] = 0
                else:
                    text_box_bitmap[y, x] = 1

    return text_box_bitmap

if __name__ == "__main__":
    box = remove_background(skimage.img_as_float(skimage.io.imread("example1.jpg")), 0, 0, 100, 50, True)
    pylab.imshow(skimage.color.gray2rgb(box))
    pylab.show()
    pass

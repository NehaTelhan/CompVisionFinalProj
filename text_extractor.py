# Split image into various scales

# Split scale images into 20 x 10 windows

# Get 48 x 48 prediction from neural network

# Save inputs in saliency map

# Run segmentation on saliency map

# Run color histogram for text boxes returned from segmentation

# Adjust resolution of text boxes

# Remove backgrounds

# Binarize

# Done!

# Import some libraries
import numpy, math, sys, glob
import skimage.io, skimage.transform, pylab
import scipy.ndimage.filters, scipy.signal, scipy.misc

from window_divider import divide_picture_to_windows, convertWindowToArray
from train_neural_network import neural_network_predict_filename, neural_network_predict, load_cnn_model


def get_gaussian_pyramid(input_image):

    height_of_window = 48
    width_of_window = 48
    images = []
    images.append(input_image)

    rows = input_image.shape[0]
    cols = input_image.shape[1]

    while int((2 / 3) * rows) > height_of_window and int((2 / 3) * cols) > width_of_window:

        resized_image = skimage.transform.resize(input_image, (int((2 / 3) * rows), int((2 / 3) * cols), 3))
        images.append(resized_image)
        rows = int((2 / 3) * rows)
        cols = int((2 / 3) * cols)

    return images

def get_windows(image):
    images = []
    windows_for_image = divide_picture_to_windows(image)

    for j in windows_for_image:
        images.append(j)

    return images

def saliency_map(image_file):
    # Read image from filename
    A = skimage.io.imread(image_file)

    # Convert image to float
    # A = skimage.img_as_float(A)

    # Only for images from dataset
    # A = skimage.transform.resize(A, (48, 48, 3))

    # Split image into various scales
    image_list = get_gaussian_pyramid(A)
    print("len image_list", len(image_list))

    # Initialize empty saliency map
    saliency_map = numpy.zeros((A.shape[0], A.shape[1]))

    # Load model
    model = load_cnn_model("modelcnn.h5")

    # Iterate through image scales
    for image in image_list:
        scale = 1.0
        windows = get_windows(image)
        # print("Window length:", len(windows))

        image_height = image.shape[0]
        image_width = image.shape[1]
        print("len windows", len(windows))
        start_pixel = [0, 0]  # (row, column)

        start_pixel = [0, 0]  # (row, column)

        for window in windows:
            result = neural_network_predict(window, model)
            print("result:", result)

            for i in range(start_pixel[0], start_pixel[0] + int(48 * scale)):
                for j in range(start_pixel[1], start_pixel[1] + int(48 * scale)):
                    if result >= 0.5:
                        saliency_map[i][j] += result

            if start_pixel[0] + 48 < image_height:
                start_pixel[0] += 48
            elif start_pixel[1] + 48 < image_width:
                start_pixel[1] += 48

        scale *= 1.5

    # Extract initial text boxes
    saliency_checked = numpy.zeros(saliency_map.shape)
    print("SALIENCY:", saliency_map)

    boxes = []
    for i in range(saliency_map.shape[0]):
        for j in range(saliency_map.shape[1]):
            if saliency_checked[i][j] == 0:
                saliency_checked[i][j] = 1
                if saliency_map[i][j] >= 0.0:
                    box = (i, j, 1, 1)  # (row, col, width, height)
                    old_box = (0, 0, 0, 0)

                    th_region = 0.0

                    while box is not old_box:
                        old_box = box

                        if box[0] - 1 >= 0:
                            sum = 0.0
                            for ii in range(box[2]):
                                print(saliency_map[box[0]-1][box[1]+ii])
                                sum += saliency_map[box[0]-1][box[1]+ii]
                            print("case 1:", sum)
                            sum /= box[2]

                            if sum > th_region:
                                for ii in range(box[2]):
                                    saliency_checked[box[0]-1][box[1]+ii] = 1
                                box[0] -= 1
                                box[3] += 1

                        if box[1] - 1 >= 0:
                            sum = 0.0
                            for ii in range(box[3]):
                                sum += saliency_map[box[0]+ii][box[1]-1]
                            print("case 2:", sum)
                            sum /= box[3]


                            if sum > th_region:
                                for ii in range(box[3]):
                                    saliency_checked[box[0]+ii][box[1]-1] = 1
                                box[1] -= 1
                                box[2] += 1

                        if box[0] + 1 < saliency_map.shape[0]:
                            sum = 0.0
                            for ii in range(box[2]):
                                sum += saliency_map[box[0]+1][box[1]+ii]
                            print("case 3:", sum)

                            sum /= box[2]

                            if sum > th_region:
                                for ii in range(box[2]):
                                    saliency_checked[box[0]+1][box[1]+ii] = 1
                                box[3] += 1

                        if box[1] + 1 < saliency_map.shape[1]:
                            sum = 0.0
                            for ii in range(box[3]):
                                sum += saliency_map[box[0]+ii][box[1]+1]
                            print("case 4:", sum)
                            sum /= box[3]

                            if sum > th_region:
                                for ii in range(box[3]):
                                    saliency_checked[box[0]+ii][box[1]+1] = 1
                                box[2] += 1

                    boxes.append(box)
    return [boxes, saliency_map]

if __name__ == "__main__":
    print("please pycharm stop giving me this error")
    saliency_map("demo_image.jpg")
    # Image cmd line param
    # image_file = sys.argv[1]
    #
    # result = neural_network_predict_filename(image_file)
    #
    # max = result[0][0]
    # index = 0
    # for i in range(len(result[0])):
    #     if result[0][i] > max:
    #         index = i
    #         max = result[0][i]
    # print(index, max)
    # print(result)


    #
    # # Read image from filename
    # A = skimage.io.imread(image_file, True)
    #
    # # Convert image to float
    # A = skimage.img_as_float(A)
    #
    # A = skimage.transform.resize(A, (48, 48, 3))
    #
    # # Split image into various scales
    # image_list = get_gaussian_pyramid(A)
    #
    # # Initialize empty saliency map
    # saliency_map = numpy.zeros((A.shape[0], A.shape[1]))
    #
    # # Iterate through image scales
    # for image in image_list:
    #     scale = 1.0
    #     windows = get_windows(image)
    #
    #     image_height = image.shape[0]
    #     image_width = image.shape[1]
    #
    #     start_pixel = (0, 0)  # (row, column)
    #
    #     for window in windows:
    #         result = neural_network_predict(window)  # FIXME, no need to resize
    #
    #         print (result)
    #
    #         for i in range(start_pixel[0], start_pixel[0] + int(48 * scale)):
    #             for j in range(start_pixel[1], start_pixel[1] + int(48 * scale)):
    #                 if result > -1:
    #                     saliency_map[i][j] += result
    #                     print("SALIENCY", saliency_map[i, j])
    #
    #         if start_pixel[0] + 48 < image_height:
    #             start_pixel[0] += 48
    #         elif start_pixel[1] + 48 < image_width:
    #             start_pixel[1] += 48
    #
    #     scale *= 1.5
    #
    # # Extract initial text boxes
    # saliency_checked = numpy.zeros(saliency_map.shape)
    # print("SALIENCY:", saliency_map)
    #
    #
    # boxes = []
    # for i in range(saliency_map.shape[0]):
    #     for j in range(saliency_map.shape[1]):
    #         if saliency_checked[i][j] == 0:
    #             saliency_checked[i][j] = 1
    #             if saliency_map[i][j] >= 0.0:
    #                 box = (i, j, 1, 1)  # (row, col, width, height)
    #                 old_box = (0, 0, 0, 0)
    #
    #                 th_region = 0.0
    #
    #                 while box is not old_box:
    #                     old_box = box
    #
    #                     if box[0] - 1 >= 0:
    #                         sum = 0.0
    #                         for ii in range(box[2]):
    #                             print(saliency_map[box[0]-1][box[1]+ii])
    #                             sum += saliency_map[box[0]-1][box[1]+ii]
    #                         print("case 1:", sum)
    #                         sum /= box[2]
    #
    #                         if sum > th_region:
    #                             for ii in range(box[2]):
    #                                 saliency_checked[box[0]-1][box[1]+ii] = 1
    #                             box[0] -= 1
    #                             box[3] += 1
    #
    #                     if box[1] - 1 >= 0:
    #                         sum = 0.0
    #                         for ii in range(box[3]):
    #                             sum += saliency_map[box[0]+ii][box[1]-1]
    #                         print("case 2:", sum)
    #                         sum /= box[3]
    #
    #
    #                         if sum > th_region:
    #                             for ii in range(box[3]):
    #                                 saliency_checked[box[0]+ii][box[1]-1] = 1
    #                             box[1] -= 1
    #                             box[2] += 1
    #
    #                     if box[0] + 1 < saliency_map.shape[0]:
    #                         sum = 0.0
    #                         for ii in range(box[2]):
    #                             sum += saliency_map[box[0]+1][box[1]+ii]
    #                         print("case 3:", sum)
    #
    #                         sum /= box[2]
    #
    #                         if sum > th_region:
    #                             for ii in range(box[2]):
    #                                 saliency_checked[box[0]+1][box[1]+ii] = 1
    #                             box[3] += 1
    #
    #                     if box[1] + 1 < saliency_map.shape[1]:
    #                         sum = 0.0
    #                         for ii in range(box[3]):
    #                             sum += saliency_map[box[0]+ii][box[1]+1]
    #                         print("case 4:", sum)
    #                         sum /= box[3]
    #
    #                         if sum > th_region:
    #                             for ii in range(box[3]):
    #                                 saliency_checked[box[0]+ii][box[1]+1] = 1
    #                             box[2] += 1
    #
    #                 boxes.append(box)
    # print(boxes)
    #
    #




    # windows = get_windows(resized_image)
    # pylab.imshow(windows[1], cmap="gray")
    # pylab.show()
    #

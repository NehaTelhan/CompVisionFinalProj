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
import skimage.io, skimage.transform, pylab, skimage.color
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
    A = skimage.transform.rescale(A, 0.25)
    pylab.imshow(A)
    pylab.show()

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

    image_height = A.shape[0]
    image_width = A.shape[1]

    # Iterate through image scales
    b_count = 0
    scale = 1.0
    for image in image_list:
        b_count += 1
        windows = get_windows(image)
        # print("Window length:", len(windows))

        print("len windows", len(windows))
        start_pixel = [0, 0]  # (row, column) (height, width)

        for window in windows:
            result = neural_network_predict(window, model)
            # pylab.imshow(window)
            # pylab.show()
            print("result:", result)

            for i in range(start_pixel[0], start_pixel[0] + int(scale * 48)):
                for j in range(start_pixel[1], start_pixel[1] + int(scale * 48)):
                    if result >= 0.5:
                        # print("scale", scale)
                        # print("i: ", i, " j: ", j)
                        saliency_map[i][j] += result

            if start_pixel[1] + 2 * int(scale * 48) <= image_width:
                start_pixel[1] += int(scale * 48)
            elif start_pixel[0] + 2 * int(scale * 48) <= image_height:
                start_pixel[0] += int(scale * 48)
                start_pixel[1] = 0

        scale *= 1.5

        # if b_count == 1:
        #     break

    pylab.imshow(saliency_map, cmap='gray')
    pylab.show()
    print(saliency_map)
    quit()

    # Extract initial text boxes
    saliency_checked = numpy.zeros(saliency_map.shape)
    print("SALIENCY:", saliency_map)

    boxes = []
    for i in range(saliency_map.shape[0]):
        for j in range(saliency_map.shape[1]):
            if saliency_checked[i][j] == 0:
                print(i, j)
                saliency_checked[i][j] = 1
                if saliency_map[i][j] >= 0.5:
                    box = [i, j, 1, 1]  # (row, col, width, height)
                    old_box = [0, 0, 0, 0]

                    th_region = 0.5

                    while not (old_box[0] == box[0] and old_box[1] == box[1] and old_box[2] == box[2] and old_box[3] == box[3]):
                        old_box = []
                        old_box.append(box[0])
                        old_box.append(box[1])
                        old_box.append(box[2])
                        old_box.append(box[3])

                        if box[0] - 1 >= 0:
                            sum = 0.0
                            for ii in range(box[2]):
                                #print(saliency_map[box[0]-1][box[1]+ii])
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

                        if box[0] + box[3] < saliency_map.shape[0]:
                            sum = 0.0
                            for ii in range(box[2]):
                                sum += saliency_map[box[0]+box[3]][box[1]+ii]
                            print("case 3:", sum)

                            sum /= box[2]

                            if sum > th_region:
                                for ii in range(box[2]):
                                    saliency_checked[box[0]+box[3]][box[1]+ii] = 1
                                box[3] += 1

                        if box[1] + box[2] < saliency_map.shape[1]:
                            sum = 0.0
                            for ii in range(box[3]):
                                sum += saliency_map[box[0]+ii][box[1]+box[2]]
                            print("case 4:", sum)
                            sum /= box[3]

                            if sum > th_region:
                                for ii in range(box[3]):
                                    saliency_checked[box[0]+ii][box[1]+box[2]] = 1
                                box[2] += 1

                        #print("Old:", old_box)
                        #print("New:", box)

                    print("added box", box)
                    boxes.append(box)
    return [boxes, saliency_map]

if __name__ == "__main__":
    print("please pycharm stop giving me this error")

    result = saliency_map("demo-image1.jpg")

    pylab.imshow(skimage.color.rgb2gray(result[1]))
    pylab.show()
    print(result[1])
    print(result[0])

# Import some libraries
import numpy, math, sys, glob
import skimage.io, skimage.transform, pylab, skimage.color
import scipy.ndimage.filters, scipy.signal, scipy.misc

from window_divider import divide_picture_to_windows, convertWindowToArray
from train_neural_network import neural_network_predict_filename, neural_network_predict, load_cnn_model
from segmentation import run_segmentation
from color_histogram import color_histogram
from text_bitmap_isolater import remove_background, binarize_bitmap
from canny_edge import canny_edges
import timeit

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


def saliency_map(A):
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

    #pylab.imshow(saliency_map, cmap='gray')
    #pylab.show()

    # Extract initial text boxes
    saliency_checked = numpy.zeros(saliency_map.shape)
    print("SALIENCY:", saliency_map)

    boxes = []
    for i in range(saliency_map.shape[0]):
        for j in range(saliency_map.shape[1]):
            if saliency_checked[i][j] == 0:
                print(i, j)
                saliency_checked[i][j] = 1
                if saliency_map[i][j] >= 3:
                    box = [i, j, 1, 1]  # (row, col, width, height)
                    old_box = [0, 0, 0, 0]

                    th_region = 2.7

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

def combine_boxes(boxes, original_height, original_width):
    for j in range(len(boxes)):
        box = boxes.pop(0)
        boxes_to_remove = []
        for b in boxes:
            shared = share_pixels(box, b, original_height, original_width)
            if shared:
                highest_row = box[0]
                highest_col = box[1]
                lowest_row = box[1] + box[2]
                lowest_col = box[0] + box[3]
                if b[0] < highest_row:
                    highest_row = b[0]
                if b[1] < highest_col:
                    highest_col = b[1]
                if b[0] + b[3] > lowest_row:
                    lowest_row = b[0] + b[3]
                if b[1] + b[2] > lowest_col:
                    lowest_col = b[1] + b[2]
                box = [highest_row, highest_col, lowest_col - highest_col, lowest_row - highest_row]
                boxes_to_remove.append(b)
        for b in boxes_to_remove:
            boxes.remove(b)
        boxes.append(box)
    return boxes

def share_pixels(box1, box2, original_height, original_width):
    img = numpy.zeros((original_height, original_width))
    for i in range(box1[0], box1[0] + box1[3]):
        for j in range(box1[1], box1[1] + box1[2]):
            img[i, j] = 1
    shared = False
    for i in range(box2[0], box2[0] + box2[3]):
        for j in range(box2[1], box2[1] + box2[2]):
            if img[i, j] == 1:
                shared = True
    return shared

if __name__ == "__main__":

    start_time = timeit.default_timer()
    # Read in original image
    image_file = "demo-image1.jpg"
    # image_file = "training_set/13.jpg"

    A = skimage.io.imread(image_file)

    A = skimage.transform.rescale(A, 0.25) # FOR demo-image1 ONLY
    # edge_image = canny_edges(A)
    #
    #
    # # Saliency map
    #result = saliency_map(A)
    # pylab.imshow(result[1], cmap="gray")
    # pylab.show()
    edge_image = canny_edges(A)
    #pylab.imshow(edge_image)
    #pylab.show()

    #results = saliency_map(A)
    # print(result[0])
    results = numpy.ndarray.tolist(numpy.load("saliency_boxes.npy"))
    #pylab.imshow(results[1], cmap="gray")
    #pylab.show()

    # Run segmentation on saliency map
    # Get edge image for segmentation
    #print(results[0])
    #exit()


    boxes = run_segmentation(combine_boxes(results, A.shape[0], A.shape[1]), edge_image)
    # numpy.save("saliency_boxes1.npy", numpy.array(boxes))
    print(boxes)


    # Process each text box
    for box in boxes:
        img = numpy.zeros((box[3], box[2], 3))
        for i in range(box[3]):
            for j in range(box[2]):
                img[i, j] = A[i + box[0], j + box[1]]
        print("Sadiyah, pay attention and save one of these. Love, Your Past Self")
        pylab.imshow(img, cmap="gray")
        pylab.show()

        # Run color histogram for text boxes returned from segmentation
        is_text_inverse = color_histogram(image_file, box)
        #is_text_inverse = True

        # Remove backgrounds
        # box=(row, col, width, height)
        bitmap = remove_background(A, box[0], box[1], box[2], box[3], is_text_inverse)

        # Resize to be bigger
        bitmap = skimage.transform.rescale(bitmap, 2)

        # Done!
        pylab.imshow(skimage.color.gray2rgb(bitmap))
        pylab.show()

    print(timeit.default_timer() - start_time)

import numpy
import scipy.misc

# Expected input: Skimage.img_as_float in grayscale from gaussian pyramid
def divide_picture_to_windows(picture):
    height = picture.shape[0]
    width = picture.shape[1]

    x_step = 3
    y_step = 2

    height_of_window = 10
    width_of_window = 20

    list_of_windows = []
    count = 0

    for y in range(0, height - height_of_window, y_step):
        for x in range(0, width - width_of_window, x_step):
            count = count + 1
            window = numpy.zeros((height_of_window, width_of_window))
            for j in range(height_of_window):
                for i in range(width_of_window):
                    window[j, i] = picture[y, x]
            list_of_windows.append(window)

            # Save picture
            scipy.misc.imsave("windows/window" + str(count), thinned_image)

    windows = numpy.zeros((count, height, width))
    for i in range(count):
        windows[i] = list_of_windows[i]

    return windows

def convertWindowToArray(window):
    array = numpy.zeros(200)
    count = 0
    for y in range(10):
        for x in range(20):
            array[count] = window[y, x]
            count = count + 1
    return array

if __name__ == "__main__":
    pass
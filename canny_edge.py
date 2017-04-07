# John Mourad (jmm5kg)
# Canny Edge Detector

# Import some libraries
import numpy, skimage.io, pylab, math, sys
import scipy.ndimage.filters, scipy.signal, scipy.misc

img_file = sys.argv[1]
save_file = sys.argv[2]

#  Load an image
A = skimage.io.imread(img_file, True)

# Convert image to float
A = skimage.img_as_float(A)

# Build Gaussian filter
gauss_dim = 15
gauss_row = scipy.signal.gaussian(gauss_dim, 1)
gauss_filter = numpy.array([[gauss_row[x] * gauss_row[y] for x in range(gauss_dim)] for y in range(gauss_dim)])


# Convolve our image with the Gaussian filter
C = scipy.ndimage.filters.convolve(A, gauss_filter)

# Compute the gradient
gradient = numpy.gradient(C)
x_gradient = gradient[0]
y_gradient = gradient[1]

height = gradient[0].shape[0]
width = gradient[0].shape[1]

# Initialize magnitude and direction arrays
magnitudes = numpy.array([[0.0 for w in range(width)] for h in range(height)])
directions = numpy.array([[0.0 for w in range(width)] for h in range(height)])

# PI constant
pi = math.pi

# Compute magnitude and direction of gradient
for i in range(height):
    for j in range(width):
        mag = math.sqrt((x_gradient[i][j] ** 2) + (y_gradient[i][j] ** 2))
        direction = math.atan2(y_gradient[i][j], x_gradient[i][j])
        if direction < 0:
            direction = direction + pi
        magnitudes[i][j] = mag
        directions[i][j] = direction

# Categorize direction of gradient
for i in range(height):
    for j in range(width):
        if 0 <= directions[i][j] < (pi / 8):
            directions[i][j] = 0
        if (pi / 8) <= directions[i][j] < (3*pi / 8):
            directions[i][j] = (pi / 4)
        if (3*pi / 8) <= directions[i][j] < (5*pi / 8):
            directions[i][j] = (pi / 2)
        if (5*pi / 8) <= directions[i][j] < (7*pi / 8):
            directions[i][j] = (3*pi / 4)
        if (7*pi / 8) <= directions[i][j] <= pi:
            directions[i][j] = pi

# Create copy of magnitudes for thinned image
thinned_image = numpy.copy(magnitudes)

# Non-maximum suppression
for i in range(1, height - 1):
    for j in range(1, width - 1):
        if directions[i][j] == 0 or directions[i][j] == pi:
            if magnitudes[i][j] < magnitudes[i - 1][j] or magnitudes[i][j] < magnitudes[i + 1][j]:
                thinned_image[i][j] = 0
        if directions[i][j] == (pi/4):
            if magnitudes[i][j] < magnitudes[i - 1][j - 1] or magnitudes[i][j] < magnitudes[i + 1][j + 1]:
                thinned_image[i][j] = 0
        if directions[i][j] == (pi/2):
            if magnitudes[i][j] < magnitudes[i][j - 1] or magnitudes[i][j] < magnitudes[i][j + 1]:
                thinned_image[i][j] = 0
        if directions[i][j] == (3*pi / 4):
            if magnitudes[i][j] < magnitudes[i - 1][j + 1] or magnitudes[i][j] < magnitudes[i + 1][j - 1]:
                thinned_image[i][j] = 0

# Establish high and low thresholds
# flower.jpg thresholds
t_low = 0.1
t_high = 0.5

# building.jpg thresholds
# t_low = 0.2
# t_high = 0.5

# Initialize status array
status = numpy.array([["" for w in range(width)] for h in range(height)])

# Initialize array to hold strong edge pixels
strong = []

# Hysteresis thresholding
for i in range(height):
    for j in range(width):
        if thinned_image[i][j] < t_low:
            status[i][j] = "N"  # not strong edge
            thinned_image[i][j] = 0
        if thinned_image[i][j] > t_high:
            status[i][j] = "S"  # strong edge
            thinned_image[i][j] = 1
            strong.append((i, j))
        if t_low <= thinned_image[i][j] <= t_high:
            status[i][j] = "W"  # weak edge

# Iterative DFS for connected neighbor algorithm
visited = {}

while len(strong) > 0:
    v = strong.pop()  # pop from stack

    if v not in visited:
        visited[v] = 1  # mark as visited

        # if a weak edge, mark as strong
        if status[v[0]][v[1]] == "W":
            status[v[0]][v[1]] = "S"
            thinned_image[v[0]][v[1]] = 1

        # visit bottom row
        if 0 <= v[0] + 1 < height and 0 <= v[1] + 1 < width:
            if (v[0]+1, v[1]+1) not in visited:
                strong.append((v[0]+1, v[1]+1))

        if 0 <= v[0] + 1 < height and 0 <= v[1] < width:
            if (v[0]+1, v[1]) not in visited:
                strong.append((v[0]+1, v[1]))

        if 0 <= v[0] + 1 < height and 0 <= v[1] - 1 < width:
            if (v[0]+1, v[1] - 1) not in visited:
                strong.append((v[0]+1, v[1] - 1))

        # visit middle row
        if 0 <= v[0] < height and 0 <= v[1] + 1 < width:
            if (v[0], v[1]+1) not in visited:
                strong.append((v[0], v[1]+1))

        if 0 <= v[0] < height and 0 <= v[1] - 1 < width:
            if (v[0], v[1]-1) not in visited:
                strong.append((v[0], v[1]-1))

        # visit top row
        if 0 <= v[0] - 1 < height and 0 <= v[1] + 1 < width:
            if (v[0]-1, v[1]+1) not in visited:
                strong.append((v[0]-1, v[1]+1))

        if 0 <= v[0] - 1 < height and 0 <= v[1] < width:
            if (v[0]-1, v[1]) not in visited:
                strong.append((v[0]-1, v[1]))

        if 0 <= v[0] - 1 < height and 0 <= v[1] - 1 < width:
            if (v[0]-1, v[1] - 1) not in visited:
                strong.append((v[0]-1, v[1] - 1))

# Mark any remaining weak edges as no edges
for i in range(height):
    for j in range(width):
        if status[i][j] == "W":
            status[i][j] = "N"
            thinned_image[i][j] = 0


# Show the result
# pylab.imshow(thinned_image, cmap="gray")
# pylab.show()

scipy.misc.imsave(save_file, thinned_image)
import os
import random

image_filenames = []
for filename in os.listdir('../dataset'):
    image_filenames.append(filename)

training = []
testing = []

for i in range(len(image_filenames)):
    rand = random.random()

    if rand < 0.95:
        training.append(image_filenames[i])
    else:
        testing.append(image_filenames[i])

for train in training:
    os.rename("../dataset/{0}".format(train), "../training/{0}".format(train))

for test in testing:
    os.rename("../dataset/{0}".format(test), "../testing/{0}".format(test))


print(len(training))
print(len(testing))
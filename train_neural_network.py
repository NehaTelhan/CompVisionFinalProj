from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras
import skimage.io
import numpy
import window_divider
import read_in_classifications
import random


def read_image_list():
    with open("demo-finished.txt", "r") as images_list:
        images = []
        classifications = []
        for i in images_list:
            input_image = skimage.img_as_float(skimage.io.imread("../edges/" + i.replace("\n", ""), True))
            windows_for_image = window_divider.divide_picture_to_windows(input_image)
            for j in windows_for_image:
                images.append(window_divider.convertWindowToArray(j))
            classifications.append(read_in_classifications.retrieve_classifications("classifications_train_unbalance.txt", i))
            print(i)
        return [numpy.array(images), numpy.array(classifications)]


def get_classifications_count(classifications):
    no_text = 0
    text = 0
    for c in classifications:
        print(c)
        if c == 0:
            no_text += 1
        elif c == 1:
            text += 1
        else:
            raise ValueError
    return no_text, text


def seperate_data(images, classifications):
    training = []
    validation = []

    training_classifications = []
    validation_classifications = []

    for i in range(len(images)):
        rand = random.random()

        if rand < 0.67:
            training.append(i)
            training_classifications.append(classifications[i])
        else:
            validation.append(i)
            validation_classifications.append(classifications[i])
    return numpy.array(training), numpy.array(validation), numpy.array(training_classifications), numpy.array(validation_classifications)


# Training and validation images should be output consolidated from window_divider
def train_network(training_images, training_classifications, validation_images, validation_classifications):
    # TODO: Decide on how many images per batch, how many epochs, and number of samples
    batch_size = 50000
    epochs = 3

    num_classes = 2

    # Convert class vectors to binary class matrices
    train_classifications = keras.utils.to_categorical(training_classifications, num_classes)
    validation_classifications = keras.utils.to_categorical(validation_classifications, num_classes)

    # Create model
    model = Sequential()
    model.add(Dense(2, activation='tanh', input_shape=(200,)))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    model.fit(training_images, train_classifications,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(validation_images, validation_classifications))

    score = model.evaluate(validation_images, validation_classifications, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    return model

if __name__ == "__main__":
    images, classifications = read_image_list()

    training_images, validation_images, training_classifications, validation_classifications = seperate_data(images, classifications)
    train_network(training_images, training_classifications, validation_images, validation_classifications)

    pass
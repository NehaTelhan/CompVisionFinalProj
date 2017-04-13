from keras.layers import Dense
from keras.models import Sequential, model_from_json
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
            input_image = skimage.img_as_float(skimage.io.imread("flower.jpg", True))
            windows_for_image = window_divider.divide_picture_to_windows(input_image)
            for j in windows_for_image:
                images.append(window_divider.convertWindowToArray(j))
                print(len(window_divider.convertWindowToArray(j)))
            classifications.append(read_in_classifications.retrieve_classifications("classifications_train_unbalance.txt", i))
            print(i)
            break
        print(len(numpy.array(images)[0]))
        numpy.save("test.npy", numpy.array(images))
        print(len(numpy.load("test.npy")[0]))
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
    batch_size = 100000
    epochs = 1

    num_classes = 2

    # Convert class vectors to binary class matrices
    #train_classifications = keras.utils.to_categorical(training_classifications, num_classes)
    #validation_classifications = keras.utils.to_categorical(validation_classifications, num_classes)

    # Create model
    model = Sequential()
    model.add(Dense(2, activation='tanh', input_shape=(200,)))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    model.fit(training_images, training_classifications,
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

def neural_network_predict(window):
    window_input = numpy.zeros((1, 200))
    window_input[0] = window

    json_file = open('model.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model.predict(window_input)[0]

def read_in_data(training_images_filename, training_classifications_filename, validation_images_filename, validation_classifications_filename):
    return numpy.load(training_images_filename), numpy.load(training_classifications_filename), numpy.load(validation_images_filename), numpy.load(validation_classifications_filename)

if __name__ == "__main__":
    #images, classifications = read_image_list()

    training_images, training_classifications, validation_images, validation_classifications = read_in_data("training_images.npy", "training_classifications.npy", "validation_images.npy", "validation_classifications.npy")
    train_network(training_images, training_classifications, validation_images, validation_classifications)

    pass
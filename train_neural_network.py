from keras.layers import Dense, Flatten, Input
from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.initializers import VarianceScaling, Ones
from keras import optimizers
from keras.applications import VGG16
import skimage.io, skimage.transform
import numpy
import window_divider
import read_in_classifications
import random
import pylab

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


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
    batch_size = 16
    epochs = 1

    num_classes = 2

    # Convert class vectors to binary class matrices
    #train_classifications = keras.utils.to_categorical(training_classifications, num_classes)
    #validation_classifications = keras.utils.to_categorical(validation_classifications, num_classes)

    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    # Add an additional MLP model at the "top" (end) of the network
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    # top_model.add(Dense(1, activation='tanh'))
    # model = Model(input=base_model.input, output=top_model(base_model.output))
    #
    # # model = load_model("model.h5")
    # # Freeze all the layers in the original model (fine-tune only the added Dense layers)
    # for layer in model.layers[:19]:  # You need to figure out how many layers were in the base model to freeze
    #     layer.trainable = False

    # model = Sequential()
    # model.add(Conv2D(96, kernel_size=(7, 7), activation='relu', input_shape=(48, 48, 3), kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # # model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='relu', kernel_initializer=VarianceScaling(), use_bias=True))
    # model.add(Dense(2, activation="softmax"))

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(48,48,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    # Compile the model with a SGD/momentum optimizer and a slow learning rate.
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(training_images, training_classifications,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(validation_images, validation_classifications))

    score = model.evaluate(validation_images, validation_classifications, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model-v2.h5")
    model.save("model.h5")

    return model


def neural_network_predict(image):
    image_input = numpy.zeros((1, 48, 48, 3))
    image_input[0] = image

    model = load_model("model.h5")

    model.compile(loss="mean_squared_error",
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    return model.predict(image_input)[0]


def neural_network_predict_filename(filename):

    A = skimage.io.imread(filename)
    A = skimage.img_as_float(A)
    A = skimage.transform.resize(A, (224, 224, 3))
    pylab.imshow(A)
    pylab.show()

    image_input = numpy.zeros((1, 224, 224, 3))
    image_input[0] = A

    base_model = VGG16(weights='imagenet', include_top=True)

    base_model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return base_model.predict(image_input)


def read_in_data(training_images_filename, training_classifications_filename, validation_images_filename, validation_classifications_filename):
    # return numpy.load(training_images_filename), numpy.load(training_classifications_filename), numpy.load(validation_images_filename), numpy.load(validation_classifications_filename)
    training_images = []
    training_classifications = []
    validation_images = []
    validation_classifications = []
    train_text_images = 0
    val_text_images = 0
    train_non_images = 0
    val_non_images = 0
    for i in range(20000):
        filename = "txt_" + str(i) + ".jpg"
        try:
            image = skimage.transform.resize(skimage.io.imread("../training/" + filename), (48, 48, 3))
            if i >= 680674:
                classification = read_in_classifications.retrieve_classifications("classifications_val_unbalance.txt", filename)
            else:
                classification = read_in_classifications.retrieve_classifications("classifications_train_unbalance.txt", filename)

            if classification == 1:
                print(filename, classification)
                train_text_images += 1
                training_classifications.append(classification)
                training_images.append(image)
            elif train_non_images < train_text_images:
                print(filename, classification)
                train_non_images += 1
                training_classifications.append(classification)
                training_images.append(image)

        except FileNotFoundError:
            image = skimage.transform.resize(skimage.io.imread("../testing/" + filename), (48, 48, 3))
            if i >= 680674:
                classification = read_in_classifications.retrieve_classifications("classifications_val_unbalance.txt", filename)
            else:
                classification = read_in_classifications.retrieve_classifications("classifications_train_unbalance.txt", filename)

            if classification == 1:
                print(filename, classification)
                val_text_images += 1
                validation_classifications.append(classification)
                validation_images.append(image)
            elif val_non_images < val_text_images:
                print(filename, classification)
                val_non_images += 1
                validation_classifications.append(classification)
                validation_images.append(image)

    return numpy.array(training_images), numpy.array(training_classifications), numpy.array(validation_images), numpy.array(validation_classifications)

if __name__ == "__main__":
    #images, classifications = read_image_list()

    training_images, training_classifications, validation_images, validation_classifications = read_in_data("training_data/training_images3.npy", "training_data/training_classifications3.npy", "training_data/validation_images3.npy", "training_data/validation_classifications3.npy")

    print(training_images.shape, training_classifications.shape, validation_images.shape, validation_classifications.shape)

    train_network(training_images, training_classifications, validation_images, validation_classifications)

    pass
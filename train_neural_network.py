from keras.layers import Dense, Flatten, Input
from keras.models import Sequential, model_from_json, load_model, Model
from keras.optimizers import RMSprop
from keras import optimizers
import skimage.io
import numpy
import window_divider
import read_in_classifications
import random

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
def train_network():
    # TODO: Decide on how many images per batch, how many epochs, and number of samples
    # batch_size = 50
    # epochs = 1
    #
    # num_classes = 2
    #
    # # Convert class vectors to binary class matrices
    # #train_classifications = keras.utils.to_categorical(training_classifications, num_classes)
    # #validation_classifications = keras.utils.to_categorical(validation_classifications, num_classes)
    #
    # # Create model
    # # Replace with newly loaded model
    # # model = Sequential()
    # model = load_model('model.h5')
    # #model.add(Dense(2, activation='tanh', input_shape=(200,)))
    # #model.add(Dense(1, activation='tanh'))
    #
    # model.compile(loss='mean_squared_error',
    #               optimizer=RMSprop(),
    #               metrics=['accuracy'])
    #
    # model.fit(training_images, training_classifications,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=0,
    #           validation_data=(validation_images, validation_classifications))
    #
    # score = model.evaluate(validation_images, validation_classifications, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    #
    # # Save model
    # # model_json = model.to_json()
    # # with open("model.json", "w") as json_file:
    # #     json_file.write(model_json)
    # # model.save_weights("model-v2.h5")
    # model.save("model.h5")
    #
    # return model

    img_width, img_height = 48, 48
    train_data_dir = '../training'
    validation_data_dir = '../testing'
    nb_train_samples = 947765
    nb_validation_samples = 49640
    epochs = 1
    batch_size = 16

    input_tensor = Input(shape=(48, 48, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(1, activation='tanh'))
    model = Model(input=base_model.input, output=top_model(base_model.output))

    for layer in model.layers[:19]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255
    )

    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    # validation_generator = test_datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_height, img_width),
    #     batch_size=batch_size,
    #     class_mode='binary')

    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size
    )

    # model.fit_generator(
    #     train_generator,
    #     samples_per_epoch=nb_train_samples // batch_size,
    #     epochs=epochs,  # For Keras 2.0 API change to epochs=epochs,
    #     validation_data=validation_generator,
    #     validation_steps=nb_validation_samples // batch_size)

    model.save("model1.h5")


def neural_network_predict(window):
    window_input = numpy.zeros((1, 200))
    window_input[0] = window

    # json_file = open('model.json', 'r')
    # model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(model_json)
    # model.save("model.h5")
    model = load_model("model.h5")

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model.predict(window_input)[0]


def read_in_data(training_images_filename, training_classifications_filename, validation_images_filename, validation_classifications_filename):
    return numpy.load(training_images_filename), numpy.load(training_classifications_filename), numpy.load(validation_images_filename), numpy.load(validation_classifications_filename)

if __name__ == "__main__":
    #images, classifications = read_image_list()

    # training_images, training_classifications, validation_images, validation_classifications = read_in_data("training_data/training_images3.npy", "training_data/training_classifications3.npy", "training_data/validation_images3.npy", "training_data/validation_classifications3.npy")

    train_network()

    pass
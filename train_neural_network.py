from keras.layers import Dense, Flatten, Input, BatchNormalization
from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import RMSprop
from keras.initializers import Constant
from keras import optimizers
from keras.applications import VGG16
import skimage.io, skimage.transform, skimage
import numpy
import read_in_classifications
import random
import pylab
import keras
import sklearn.metrics

from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

def confusion_matrix(training_images, training_classifications):
    model = load_model("modelcnn7.h5")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1),
                  metrics=['accuracy'])

    predictions = model.predict_classes(training_images, verbose=1)
    
    matrix = sklearn.metrics.confusion_matrix(training_classifications, predictions)
    
    matrix = matrix.astype(float)
    for i in range(matrix.shape[0]):
        sum = 0.0
        for j in range(matrix.shape[1]):
            sum = sum + matrix[i, j]
        for j in range(matrix.shape[1]):
            matrix[i, j] = matrix[i, j] / sum

    numpy.save("matrix.npy", matrix)
    print(matrix)
    return matrix

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
    epochs = 10

    num_classes = 2

    # Convert class vectors to binary class matrices
    train_classifications = keras.utils.to_categorical(training_classifications, num_classes)
    # validation_classifications = keras.utils.to_categorical(validation_classifications, num_classes)

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

    model = Sequential()
    model.add(Conv2D(96, kernel_size=(7, 7),
                     activation='relu',
                     input_shape=(48, 48, 3),
                     kernel_initializer='glorot_normal',
                     use_bias=True,
                     bias_initializer='zeros', data_format='channels_last', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=1, data_format='channels_last'))
    model.add(
        BatchNormalization(scale=True, gamma_initializer=Constant(value=0.0001), beta_initializer=Constant(value=0.75)))
    model.add(Conv2D(256, kernel_size=(5, 5),
                     activation='relu',
                     kernel_initializer='glorot_normal',
                     use_bias=True,
                     bias_initializer='ones', data_format='channels_last', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last'))
    model.add(
        BatchNormalization(scale=True, gamma_initializer=Constant(value=0.0001), beta_initializer=Constant(value=0.75)))
    model.add(Conv2D(384, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='glorot_normal',
                     use_bias=True,
                     bias_initializer='zeros', data_format='channels_last', padding='same'))
    model.add(Conv2D(384, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='glorot_normal',
                     use_bias=True,
                     bias_initializer='ones', data_format='channels_last', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3),
                     activation='relu',
                     kernel_initializer='glorot_normal',
                     use_bias=True,
                     bias_initializer='ones', data_format='channels_last', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu',
                    kernel_initializer='glorot_normal',
                    use_bias=True,
                    bias_initializer='ones'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',
                    kernel_initializer='glorot_normal',
                    use_bias=True,
                    bias_initializer='ones'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer='glorot_normal',
                    use_bias=True,
                    bias_initializer='zeros'))

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape=(48, 48, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    model = Sequential()
    model.add(Conv2D(96, kernel_size=(5,5),
                     activation='relu',
                     input_shape=(48, 48, 3),
                     kernel_initializer='glorot_normal'))
    model.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    #model = load_model("modelcnn6.h5")

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(48,48,3)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='tanh'))

    # Compile the model with a SGD/momentum optimizer and a slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1),
                  metrics=['accuracy'])

    model.fit(training_images, train_classifications,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              class_weight={0: 1, 1: 1}
    )

    model.save("modelcnn200.h5")

    score = model.evaluate(training_images, train_classifications, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save model
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("model-v2.h5")

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
    A = skimage.transform.resize(A, (48, 48, 3))

    image_input = numpy.zeros((1, 48, 48, 3))
    image_input[0] = A

    base_model = load_model("modelcnn.h5")

    base_model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1),
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
    
    for i in range(1, 86):
        image = skimage.transform.resize(skimage.io.imread("Vision/char/" + str(62) + "/" + str(6100 + i) + ".jpg"), (48, 48, 3))
        classification = 1
        #train_text_images = train_text_images + 1
        #training_classifications.append(classification)
        #training_images.append(image)
        print("char/" + str(62) + "/" + str(6100 + i))

    count = 1
    for i in range(1, 62):
        for j in range(100):
            image = skimage.transform.resize(skimage.io.imread("Vision/char/" + str(i) + "/" + str(count) + ".jpg"), (48, 48, 3))
            classification = 1
            #train_text_images = train_text_images + 1
            #training_classifications.append(classification)
            #training_images.append(image)
            count = count + 1
            print("char/" + str(i) + "/" + str(count - 1))


    for i in range(0, 300000): # 300000 (cnn), 200000 (float), 100000 (mod)
        
        try:
            filename = "Vision/training/txt_" + str(i) + ".jpg"
            image = skimage.img_as_float(skimage.transform.resize(skimage.io.imread(filename), (48, 48, 3)))
        except:
            continue
            try:
                filename = "Vision/testing/txt_" + str(i) + ".jpg"
                image = skimage.transform.resize(skimage.io.imread(filename), (48, 48, 3))
            except:
                continue

        if i >= 680874:
            classification = read_in_classifications.retrieve_classifications("Vision/classifications_val_unbalance.txt", filename)
        else:
            classification = read_in_classifications.retrieve_classifications("Vision/classifications_train_unbalance.txt", filename)

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

    count = 1
    for i in range(1, 62):
        for j in range(100):
            image = skimage.transform.resize(skimage.io.imread("Vision/char/" + str(i) + "/" + str(count) + ".jpg"), (48, 48, 3))
            classification = 1
            #training_classifications.append(classification)
            #training_images.append(image)
            count = count + 1
            print("char/" + str(i) + "/" + str(count - 1))

    return numpy.array(training_images), numpy.array(training_classifications), numpy.array(validation_images), numpy.array(validation_classifications)

if __name__ == "__main__":
    #images, classifications = read_image_list()

    # training_images, training_classifications, validation_images, validation_classifications = read_in_data("training_data/training_images3.npy", "training_data/training_classifications3.npy", "training_data/validation_images3.npy", "training_data/validation_classifications3.npy")

    #print(training_images.shape, training_classifications.shape, validation_images.shape, validation_classifications.shape)

    #numpy.save("Vision/images_0_3.npy", training_images)
    #numpy.save("Vision/classifications_0_3.npy", training_classifications)

    
    training_images = numpy.load("Vision/images_0_3.npy")
    training_classifications = numpy.load("Vision/classifications_0_3.npy")
    
    training_images = numpy.append(training_images, numpy.load("Vision/images_3_35.npy"), axis=0)
    training_classifications = numpy.append(training_classifications, numpy.load("Vision/classifications_3_35.npy"))

    training_images = numpy.append(training_images, numpy.load("Vision/images_35_38.npy"), axis=0)
    training_classifications = numpy.append(training_classifications, numpy.load("Vision/classifications_35_38.npy"))

    training_images = numpy.append(training_images, numpy.load("Vision/images_5_10.npy"), axis=0)
    training_classifications = numpy.append(training_classifications, numpy.load("Vision/classifications_5_10.npy"))

    #train_network(training_images, training_classifications, [], [])

    confusion_matrix(training_images, training_classifications)

    #print(neural_network_predict_filename("Vision/char/1/1.jpg"))
    #print(neural_network_predict_filename("Vision/txt_591.jpg"))
    #print(neural_network_predict_filename("Vision/txt_1316.jpg"))
    #print(neural_network_predict_filename("Vision/txt_1375.jpg"))

    pass

from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras import optimizers
import skimage.io, skimage.transform, skimage
import numpy
import read_in_classifications
import keras
import sklearn.metrics

def load_cnn_model(modelFileName):
    model = load_model(modelFileName)

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1),
                  metrics=['accuracy'])

    return model

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

# Training and validation images should be output consolidated from window_divider
def train_network(training_images, training_classifications, validation_images, validation_classifications):
    batch_size = 16
    epochs = 10

    num_classes = 2

    # Convert class vectors to binary class matrices
    train_classifications = keras.utils.to_categorical(training_classifications, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape=(48, 48, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    #model = load_model("modelcnn7.h5")

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

    model.save("modelcnn8.h5")

    score = model.evaluate(training_images, train_classifications, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model

def neural_network_predict(image, model):
    image_input = numpy.zeros((1, 48, 48, 3))
    image_input[0] = image

    return model.predict(image_input)[0][1]

def neural_network_predict_filename(filename):

    A = skimage.io.imread(filename)
    A = skimage.img_as_float(A)
    A = skimage.transform.resize(A, (48, 48, 3))

    image_input = numpy.zeros((1, 48, 48, 3))
    image_input[0] = A

    base_model = load_model("modelcnn7.h5")

    base_model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1),
                  metrics=['accuracy'])

    return base_model.predict(image_input)

def read_in_data():
    training_images = []
    training_classifications = []
    validation_images = []
    validation_classifications = []
    train_text_images = 0
    train_non_images = 0

    for i in range(0, 1000000): # 300000 (cnn), 200000 (float), 100000 (mod)
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

    return numpy.array(training_images), numpy.array(training_classifications), numpy.array(validation_images), numpy.array(validation_classifications)

if __name__ == "__main__":
    #training_images, training_classifications, validation_images, validation_classifications = read_in_data()

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

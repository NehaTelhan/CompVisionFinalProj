from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
import keras

# Training data: 680673

# Training and validation images should be output consolidated from window_divider
def train_network(training_images, training_classifications, validation_images, validation_classifications):
    # TODO: Decide on how many images per batch, how many epochs, and number of samples
    batch_size = 16
    epochs = 2

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
    pass
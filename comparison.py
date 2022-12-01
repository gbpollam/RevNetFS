import keras
import tensorflow as tf
from keras import layers
import pandas as pd

from utils.prepare_data import prepare_data

from layers.RevLayerKeras import RevLayerKeras

tf.random.set_seed(1234)


# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format


# DEFINE GLOBAL VARIABLES
# Same labels will be reused throughout the program
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']
# The number of steps within one time segment
TIME_PERIODS = 20
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 10


def main():
    file_path = 'dataset/WISDM_ar_v1.1_raw.txt'
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path, TIME_PERIODS, STEP_DISTANCE)

    # Define the proportion to be used when splitting channels in reversible layers
    proportion = 0.5

    # Define the keras model (Net1)
    '''
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3))
    model.add(RevLayerKeras(in_channels=16, proportion=.75))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=6, activation='softmax'))
    '''

    # Define the keras model (Net2)
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(RevLayerKeras(in_channels=16, proportion=proportion))
    model.add(RevLayerKeras(in_channels=16, proportion=proportion))
    model.add(RevLayerKeras(in_channels=16, proportion=proportion))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(RevLayerKeras(in_channels=32, proportion=proportion))
    model.add(RevLayerKeras(in_channels=32, proportion=proportion))
    model.add(RevLayerKeras(in_channels=32, proportion=proportion))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=16, activation='relu'))
    model.add(keras.layers.Dense(units=6, activation='softmax'))

    # Define the keras model (Net2)
    model2 = keras.Sequential()
    model2.add(keras.Input(shape=(20, 3)))
    model2.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model2.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model2.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model2.add(keras.layers.AveragePooling1D())
    model2.add(keras.layers.GlobalAvgPool1D())
    model2.add(keras.layers.Dense(units=16, activation='relu'))
    model2.add(keras.layers.Dense(units=6, activation='softmax'))

    loss = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # model.compile(optimizer, loss=loss)
    # model.fit(x_train, y_train_hot, epochs=1, batch_size=32)

    model2.compile(optimizer, loss=loss)
    model2.fit(x_train, y_train_hot, epochs=100, batch_size=64)

    # predictions = model.predict(x_test)
    predictions = model2.predict(x_test)

    true_preds = []

    for i in range(len(predictions)):
        if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds.append(1)
        else:
            true_preds.append(0)

    print("Accuracy on test data: ", sum(true_preds) / len(true_preds))


if __name__ == "__main__":
    main()

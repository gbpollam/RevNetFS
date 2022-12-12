import keras
import tensorflow as tf
from keras import layers
import pandas as pd

from utils.prepare_data import prepare_data

from layers.RevLayerKeras import RevLayerKeras, SmallerRevLayerKeras

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
    file_path = '../dataset/WISDM_ar_v1.1_raw.txt'
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path,
                                                            TIME_PERIODS,
                                                            STEP_DISTANCE,
                                                            scaler_type='minmax')

    # Define the proportion to be used when splitting channels in reversible layers
    proportion = 0.75

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

    # Define the keras model
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(RevLayerKeras(in_channels=16, proportion=proportion))
    model.add(keras.layers.GlobalAvgPool1D())
    # model.add(keras.layers.Dense(units=16, activation='relu'))
    model.add(keras.layers.Dense(units=6, activation='softmax'))
    # PERFORMANCE:
    # (epochs = 50, batch_size=32, proportion=0.5) = 0.7934
    # (epochs = 100, batch_size=32, proportion=0.5) = 0.8003
    # (epochs = 100, batch_size=32, proportion=0.75) = 0.8202
    # (epochs = 100, batch_size=32, proportion=0.875) = 0.7944

    # Define the model with a smaller Reversible Layer
    # Number of parameters:
    # p=0.5 : 662 (160+400+102)
    # p=0.75 : 566 (160+304+102)
    # p=0.875 : 446 (160+184+102)
    model_red = keras.Sequential()
    model_red.add(keras.Input(shape=(20, 3)))
    model_red.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model_red.add(SmallerRevLayerKeras(in_channels=16, proportion=proportion))
    model_red.add(keras.layers.GlobalAvgPool1D())
    # model.add(keras.layers.Dense(units=16, activation='relu'))
    model_red.add(keras.layers.Dense(units=6, activation='softmax'))
    # PERFORMANCE:
    # (epochs = 100, batch_size=32, proportion=0.5) = 0.7756
    # (epochs = 100, batch_size=32, proportion=0.75) = 0.7718
    # (epochs = 100, batch_size=32, proportion=0.875) = 0.7656

    # Define a model with the same number of parameters as the 0.5 case (662)
    # Number of parameters = 654 (40 + 208 + 304 + 102)
    model_same75 = keras.Sequential()
    model_same75.add(keras.Input(shape=(20, 3)))
    model_same75.add(keras.layers.Conv1D(filters=4, kernel_size=3, activation='relu'))
    model_same75.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='SAME'))
    model_same75.add(SmallerRevLayerKeras(in_channels=16, proportion=proportion))
    model_same75.add(keras.layers.GlobalAvgPool1D())
    model_same75.add(keras.layers.Dense(units=6, activation='softmax'))
    # (epochs = 100, batch_size=32, proportion=0.75) = 0.7657 0.7893 0.7896

    # Define the keras model
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
    # model.fit(x_train, y_train_hot, epochs=100, batch_size=32)

    # model2.compile(optimizer, loss=loss)
    # model2.fit(x_train, y_train_hot, epochs=100, batch_size=64)

    # model_red.compile(optimizer, loss=loss)
    # model_red.fit(x_train, y_train_hot, epochs=100, batch_size=32)

    model_same75.compile(optimizer, loss=loss)
    model_same75.fit(x_train, y_train_hot, epochs=100, batch_size=32)

    # predictions = model.predict(x_test)
    # predictions = model2.predict(x_test)
    # predictions = model_red.predict(x_test)
    predictions = model_same75.predict(x_test)

    true_preds = []

    for i in range(len(predictions)):
        if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds.append(1)
        else:
            true_preds.append(0)

    print("Accuracy on test data: ", sum(true_preds) / len(true_preds))


if __name__ == "__main__":
    main()

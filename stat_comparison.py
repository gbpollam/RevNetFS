import keras
import tensorflow as tf
from keras import layers
import pandas as pd
import numpy as np

from utils.prepare_data import prepare_data

from layers.RevLayerKeras import RevLayerKeras, SmallerRevLayerKeras


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
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path,
                                                            TIME_PERIODS,
                                                            STEP_DISTANCE,
                                                            scaler_type='minmax')

    # Define the proportion to be used when splitting channels in reversible layers
    proportion1 = 0.5

    # Define the keras model
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(SmallerRevLayerKeras(in_channels=16, proportion=proportion1))
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=6, activation='softmax'))

    # Same network with larger proportion (fewer parameters)
    proportion2 = 0.875
    # p=0.875 : 446 parameters (160+184+102)
    model2 = keras.Sequential()
    model2.add(keras.Input(shape=(20, 3)))
    model2.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model2.add(SmallerRevLayerKeras(in_channels=16, proportion=proportion2))
    model2.add(keras.layers.GlobalAvgPool1D())
    model2.add(keras.layers.Dense(units=6, activation='softmax'))

    # Network with roughly the same number of parameters as the 0.5 case (662)
    # Number of parameters = 650 (60 + 304 + 184 + 102)
    model3 = keras.Sequential()
    model3.add(keras.Input(shape=(20, 3)))
    model3.add(keras.layers.Conv1D(filters=6, kernel_size=3, activation='relu'))
    model3.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='SAME'))
    model3.add(SmallerRevLayerKeras(in_channels=16, proportion=proportion2))
    model3.add(keras.layers.GlobalAvgPool1D())
    model3.add(keras.layers.Dense(units=6, activation='softmax'))

    loss = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    saved_results = np.zeros(shape=(30, 3))

    for j in range(30):
        # Train and predict for model 1
        model.compile(optimizer, loss=loss)
        model.fit(x_train, y_train_hot, epochs=100, batch_size=32)
        predictions = model.predict(x_test)
        true_preds = []
        for i in range(len(predictions)):
            if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
                true_preds.append(1)
            else:
                true_preds.append(0)
        saved_results[j, 0] = sum(true_preds) / len(true_preds)

        # Train and predict for model 2
        model2.compile(optimizer, loss=loss)
        model2.fit(x_train, y_train_hot, epochs=100, batch_size=32)
        predictions = model2.predict(x_test)
        true_preds = []
        for i in range(len(predictions)):
            if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
                true_preds.append(1)
            else:
                true_preds.append(0)
        saved_results[j, 1] = sum(true_preds) / len(true_preds)

        # Train and predict for model 3
        model3.compile(optimizer, loss=loss)
        model3.fit(x_train, y_train_hot, epochs=100, batch_size=32)
        predictions = model3.predict(x_test)
        true_preds = []
        for i in range(len(predictions)):
            if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
                true_preds.append(1)
            else:
                true_preds.append(0)
        saved_results[j, 2] = sum(true_preds) / len(true_preds)
        tf.keras.backend.clear_session()

    pd.DataFrame(saved_results).to_csv("stat_comparison_results.csv", header=["Balanced",
                                                                              "Unbalanced_less_params",
                                                                              "Unbalanced_same_params"])


if __name__ == "__main__":
    main()
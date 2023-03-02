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

    # Define Davide Quarantiello's Network (params: 10084, Acc: 0.8039178467028587)
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.Dense(units=6, activation='softmax'))

    # Define an equivalent network with Reversible Layers
    proportion = .5

    # params: 8548, Acc: 0.7951482479784366
    model_rev = keras.Sequential()
    model_rev.add(keras.Input(shape=(20, 3)))
    model_rev.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model_rev.add(keras.layers.AveragePooling1D())
    model_rev.add(SmallerRevLayerKeras(in_channels=32, proportion=proportion))
    model_rev.add(SmallerRevLayerKeras(in_channels=32, proportion=proportion))
    model_rev.add(SmallerRevLayerKeras(in_channels=32, proportion=proportion))
    model_rev.add(SmallerRevLayerKeras(in_channels=32, proportion=proportion))
    model_rev.add(keras.layers.AveragePooling1D())
    model_rev.add(keras.layers.GlobalAvgPool1D())
    model_rev.add(keras.layers.Dense(units=50, activation='relu'))
    model_rev.add(keras.layers.Dense(units=6, activation='softmax'))
    """
    # params: 10404, Acc: 0.7915417030484796
    model_rev = keras.Sequential()
    model_rev.add(keras.Input(shape=(20, 3)))
    model_rev.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model_rev.add(keras.layers.AveragePooling1D())
    model_rev.add(SmallerRevLayerKeras(in_channels=64, proportion=proportion))
    model_rev.add(keras.layers.AveragePooling1D())
    model_rev.add(keras.layers.GlobalAvgPool1D())
    model_rev.add(keras.layers.Dense(units=50, activation='relu'))
    model_rev.add(keras.layers.Dense(units=6, activation='softmax'))
    """

    loss = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer, loss=loss)
    model.fit(x_train, y_train_hot, epochs=100, batch_size=32)

    predictions = model.predict(x_test)

    model_rev.compile(optimizer, loss=loss)
    model_rev.fit(x_train, y_train_hot, epochs=100, batch_size=32)

    predictions_rev = model_rev.predict(x_test)

    true_preds = []
    true_preds_rev = []

    for i in range(len(predictions)):
        if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds.append(1)
        else:
            true_preds.append(0)
        if tf.math.argmax(predictions_rev[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds_rev.append(1)
        else:
            true_preds_rev.append(0)

    print("Accuracy on test data (Davide): ", sum(true_preds) / len(true_preds))
    print("Accuracy on test data (Rev): ", sum(true_preds_rev) / len(true_preds_rev))


if __name__ == "__main__":
    main()

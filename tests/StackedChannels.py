import keras
import tensorflow as tf
from keras import layers
import pandas as pd

from utils.prepare_data import prepare_data

from layers.RevLayerKeras import RevLayerKeras, SmallerRevLayerKeras
from layers.StackChannels import StackChannels
from layers.InvertibleDownsampling import InvertibleDownsampling

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

# Set the batch size
BATCH_SIZE = 32


def main():
    file_path = '../dataset/WISDM_ar_v1.1_raw.txt'
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path,
                                                            TIME_PERIODS,
                                                            STEP_DISTANCE,
                                                            scaler_type='minmax')

    # Proportion for RevNet channel split
    proportion = .5

    # Params = 5190
    # Test_Acc = 0.7791
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(StackChannels(copies=6))
    model.add(SmallerRevLayerKeras(in_channels=18, proportion=proportion))
    model.add(SmallerRevLayerKeras(in_channels=18, proportion=proportion))
    model.add(keras.layers.AveragePooling1D())
    model.add(StackChannels(copies=2))
    model.add(SmallerRevLayerKeras(in_channels=36, proportion=proportion))
    model.add(SmallerRevLayerKeras(in_channels=36, proportion=proportion))
    model.add(keras.layers.AveragePooling1D())
    model.add(keras.layers.GlobalAvgPool1D())
    # model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.Dense(units=6, activation='softmax'))
    """

    # Params = 2706
    # Test_Acc = 0.789666908868408 (with new preprocessing)
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(StackChannels(copies=6))
    model.add(SmallerRevLayerKeras(in_channels=18, proportion=proportion))
    model.add(keras.layers.AveragePooling1D())
    model.add(StackChannels(copies=2))
    model.add(SmallerRevLayerKeras(in_channels=36, proportion=proportion))
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=6, activation='softmax'))
    """

    # Try the i-RevNet invertible Down-sampling layers
    # Params =
    # Test Acc =
    model = keras.Sequential()
    model.add(keras.Input(shape=(20, 3)))
    model.add(InvertibleDownsampling(target_shape=(12, 6), paddings=tf.constant([[0, 0], [2, 2], [0, 0]]), bool_pad=True))
    model.add(SmallerRevLayerKeras(in_channels=6, proportion=proportion))
    model.add(SmallerRevLayerKeras(in_channels=6, proportion=proportion))
    model.add(InvertibleDownsampling(target_shape=(8, 12), paddings=tf.constant([[0, 0], [2, 2], [0, 0]]), bool_pad=True))
    model.add(SmallerRevLayerKeras(in_channels=12, proportion=proportion))
    model.add(SmallerRevLayerKeras(in_channels=12, proportion=proportion))
    model.add(keras.layers.GlobalAvgPool1D())
    model.add(keras.layers.Dense(units=6, activation='softmax'))

    loss = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(optimizer, loss=loss)
    model.fit(x_train, y_train_hot, epochs=100, batch_size=BATCH_SIZE)

    predictions = model.predict(x_test)
    true_preds = []
    for i in range(len(predictions)):
        if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds.append(1)
        else:
            true_preds.append(0)

    print("Accuracy on test data: ", sum(true_preds) / len(true_preds))


if __name__ == "__main__":
    main()

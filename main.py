import tensorflow as tf
import pandas as pd

from utils.prepare_data import prepare_data

from models.NeuralNetwork import NeuralNetwork
from layers.FCLayer import FCLayer
from layers.ActivationLayer import ActivationLayer
from layers.AvgPool1d import AvgPool1d
from layers.ConvLayer import ConvLayer
from layers.GAPLayer import GAPLayer
from layers.RevLayer import RevLayer
from layers.MutantRevLayer import MutantRevLayer

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
    x_train, y_train_hot, x_test, y_test_hot = prepare_data(file_path, TIME_PERIODS, STEP_DISTANCE, scaler_type='minmax')

    """
    model = NeuralNetwork()
    model.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=32, stride=1))
    model.add(ActivationLayer('relu'))
    model.add(AvgPool1d(window_size=2))
    model.add(ConvLayer(input_shape=(9, 32), kernel_size=3, num_filters=64, stride=1))
    model.add(ActivationLayer('relu'))
    model.add(ConvLayer(input_shape=(7, 64), kernel_size=3, num_filters=128, stride=1, padding='SAME'))
    model.add(ActivationLayer('relu'))
    model.add(AvgPool1d(window_size=2))
    model.add(GAPLayer())
    model.add(FCLayer(units=50, input_dim=64))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(units=6, input_dim=128))
    model.add(ActivationLayer('softmax'))
    """
    model = NeuralNetwork()
    model.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=32, stride=1))
    model.add(ActivationLayer('relu'))
    model.add(AvgPool1d(window_size=2))
    model.add(RevLayer(input_shape=(9, 32), proportion=.5))
    model.add(MutantRevLayer(input_shape=(9, 32), proportion=.75, new_channels=24))
    model.add(ConvLayer(input_shape=(9, 48), kernel_size=3, num_filters=64, stride=1))
    model.add(GAPLayer())
    model.add(FCLayer(units=6, input_dim=64))
    model.add(ActivationLayer('softmax'))

    model.set_loss(tf.keras.losses.BinaryCrossentropy())

    model.fit(x_train, y_train_hot, batch_size=1, epochs=10, learning_rate=0.01)

    predictions = model.predict(x_test)
    true_preds = []
    for i in range(len(predictions)):
        if tf.math.argmax(predictions[i]) == tf.math.argmax(y_test_hot[i]):
            true_preds.append(1)
        else:
            true_preds.append(0)

    print("Accuracy on test data: ", sum(true_preds)/len(true_preds))


if __name__ == "__main__":
    main()

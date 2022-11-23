import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from keras.utils import np_utils

from models.NeuralNetwork import NeuralNetwork
from layers.FCLayer import FCLayer
from layers.ActivationLayer import ActivationLayer
from layers.AvgPool1d import AvgPool1d
from layers.ConvLayer import ConvLayer
from layers.GAPLayer import GAPLayer

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


def convert_to_float(x):
    try:
        return np.float64(x)
    except:
        return np.nan


def read_data(file_path):
    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names,
                     usecols=[0, 1, 2, 3, 4, 5])
    # Last column has a ";" character which must be removed ...
    df['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    df.dropna(axis=0, how='any', inplace=True)

    df = df.sample(frac=1, random_state=1).reset_index()

    return df


def create_segments_and_labels(df, time_steps, step, label_name):
    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name].iloc[i: i + time_steps], keepdims=True)[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def normalize(df_train):
    df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
    df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
    df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
    # Round numbers
    df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    return df_train

def main():
    # Load data set containing all the data from csv
    df = read_data('dataset/WISDM_ar_v1.1_raw.txt')

    # Define column name of the label vector
    LABEL = 'ActivityEncoded'
    # Transform the labels from String to Integer via LabelEncoder
    le = preprocessing.LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    df[LABEL] = le.fit_transform(df['activity'].values.ravel())

    # Differentiate between test set and training set
    df_test = df[df['user-id'] > 28]
    df_train = df[df['user-id'] <= 28]

    df_train = normalize(df_train)
    df_test = normalize(df_test)

    x_train, y_train = create_segments_and_labels(df_train,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    x_test, y_test = create_segments_and_labels(df_test,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    # Set input & output dimensions
    num_classes = le.classes_.size

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_train.astype('float32')
    y_test = y_train.astype('float32')

    # This is for when I used just FC Network (and I needed 1d input)
    '''
    x_train = np.concatenate((x_train[:, :, 0], x_train[:, :, 1], x_train[:, :, 2]), axis=1)
    x_test = np.concatenate((x_test[:, :, 0], x_test[:, :, 1], x_test[:, :, 2]), axis=1)
    '''

    y_train_hot = np_utils.to_categorical(y_train, num_classes)
    y_test_hot = np_utils.to_categorical(y_test, num_classes)

    # Transform data into tensors
    x_train = tf.convert_to_tensor(x_train)
    y_train_hot = tf.convert_to_tensor(y_train_hot)
    x_test = tf.convert_to_tensor(x_test)
    y_test_hot = tf.convert_to_tensor(y_test_hot)

    # Again needed only when 1d input
    '''
    x_train = tf.expand_dims(x_train, axis=2)
    x_test = tf.expand_dims(x_test, axis=2)
    '''
    y_train_hot = tf.expand_dims(y_train_hot, axis=2)
    y_test_hot = tf.expand_dims(y_test_hot, axis=2)

    # Instantiate and train the model
    '''
    model = NeuralNetwork()
    model.add(FCLayer(32, 60))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(16, 32))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(6, 16))
    model.add(ActivationLayer('softmax'))
    '''

    model = NeuralNetwork()
    model.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=32, stride=1))
    model.add(ActivationLayer('relu'))
    model.add(AvgPool1d(window_size=2))
    model.add(ConvLayer(input_shape=(9, 32), kernel_size=3, num_filters=64, stride=1))
    model.add(AvgPool1d(window_size=2))
    model.add(GAPLayer())
    model.add(FCLayer(units=50, input_dim=64))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(units=6, input_dim=50))
    model.add(ActivationLayer('softmax'))

    model.set_loss(tf.keras.losses.BinaryCrossentropy())
    model.fit(x_train, y_train_hot, batch_size=32, epochs=2, learning_rate=0.1)

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

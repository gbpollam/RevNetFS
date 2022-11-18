import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from keras.utils import np_utils

from models.NeuralNetwork import NeuralNetwork
from layers.FCLayer import FCLayer
from layers.ActivationLayer import ActivationLayer

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
        label = stats.mode(df[label_name].iloc[i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


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

    df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
    df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
    df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
    # Round numbers
    df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

    x_train, y_train = create_segments_and_labels(df_train,
                                                  TIME_PERIODS,
                                                  STEP_DISTANCE,
                                                  LABEL)

    # Set input & output dimensions
    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = le.classes_.size

    input_shape = x_train.shape

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    # TODO: add other dimensions
    x_train = np.concatenate((x_train[:, :, 0], x_train[:, :, 1], x_train[:, :, 2]), axis=1)

    y_train_hot = np_utils.to_categorical(y_train, num_classes)

    # Transform data into tensors
    x_train = tf.convert_to_tensor(x_train)
    y_train_hot = tf.convert_to_tensor(y_train_hot)

    x_train = tf.expand_dims(x_train, axis=2)
    y_train_hot = tf.expand_dims(y_train_hot, axis=2)

    # Instantiate and train the model
    model = NeuralNetwork()
    model.add(FCLayer(32, 60))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(16, 32))
    model.add(ActivationLayer('relu'))
    model.add(FCLayer(6, 16))
    model.add(ActivationLayer('softmax'))

    model.set_loss(tf.keras.losses.BinaryCrossentropy())
    model.fit(x_train, y_train_hot, learning_rate=0.1)


if __name__ == "__main__":
    main()

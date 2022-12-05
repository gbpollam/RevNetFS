import numpy as np
import tensorflow as tf
from scipy import stats
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils

from utils.gauss_rank_scaler import GaussRankScaler

pd.options.mode.chained_assignment = None


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

    # This line is wrong: one shouldn't shuffle the dataset before creating the segments
    # df = df.sample(frac=1, random_state=1).reset_index()

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


def fit_minmax(df_train):
    x_max = df_train.loc[:, 'x-axis'].max()
    y_max = df_train.loc[:, 'y-axis'].max()
    z_max = df_train.loc[:, 'z-axis'].max()
    x_min = df_train.loc[:, 'x-axis'].min()
    y_min = df_train.loc[:, 'y-axis'].min()
    z_min = df_train.loc[:, 'z-axis'].min()
    df_train.loc[:, 'x-axis'] = (df_train.loc[:, 'x-axis'] - x_min)/(x_max-x_min)
    df_train.loc[:, 'y-axis'] = (df_train.loc[:, 'y-axis'] - y_min)/(y_max-y_min)
    df_train.loc[:, 'z-axis'] = (df_train.loc[:, 'z-axis'] - z_min)/(z_max-z_min)
    minmax_list = [x_max, x_min, y_max, y_min, z_max, z_min]
    return df_train, minmax_list


def minmax(df_train, minmax_list):
    df_train.loc[:, 'x-axis'] = (df_train.loc[:, 'x-axis'] - minmax_list[1]) / (minmax_list[0] - minmax_list[1])
    df_train.loc[:, 'y-axis'] = (df_train.loc[:, 'y-axis'] - minmax_list[3]) / (minmax_list[2] - minmax_list[3])
    df_train.loc[:, 'z-axis'] = (df_train.loc[:, 'z-axis'] - minmax_list[5]) / (minmax_list[4] - minmax_list[5])
    return df_train



def fit_gauss_rank(df_train):
    # Formerly minmax was used
    """
    df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
    df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
    df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
    # Round numbers
    df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    """
    # This is Gaussian Rank Scaling
    scaler = GaussRankScaler()
    df_xyz_scaled = scaler.fit_transform(df_train[['x-axis', 'y-axis', 'z-axis']])
    df_train.loc[:, 'x-axis'] = df_xyz_scaled[:, 0]
    df_train.loc[:, 'y-axis'] = df_xyz_scaled[:, 1]
    df_train.loc[:, 'z-axis'] = df_xyz_scaled[:, 2]
    return df_train, scaler


def gauss_rank(df_train, scaler):
    # This is Gaussian Rank Scaling
    df_xyz_scaled = scaler.transform(df_train[['x-axis', 'y-axis', 'z-axis']])
    df_train.loc[:, 'x-axis'] = df_xyz_scaled[:, 0]
    df_train.loc[:, 'y-axis'] = df_xyz_scaled[:, 1]
    df_train.loc[:, 'z-axis'] = df_xyz_scaled[:, 2]
    return df_train


def prepare_data(file_path, time_periods, step_distance, scaler_type):
    # Load data set containing all the data from csv
    df = read_data(file_path)

    # Define column name of the label vector
    LABEL = 'ActivityEncoded'
    # Transform the labels from String to Integer via LabelEncoder
    le = preprocessing.LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    df[LABEL] = le.fit_transform(df['activity'].values.ravel())

    # Differentiate between test set and training set
    df_test = df[df['user-id'] > 28]
    df_train = df[df['user-id'] <= 28]

    if scaler_type == 'minmax':
        df_train, scaler = fit_minmax(df_train)
        df_test = minmax(df_test, scaler)
    elif scaler_type == 'gauss_rank':
        df_train, scaler = fit_gauss_rank(df_train)
        df_test = gauss_rank(df_test, scaler)
    else:
        raise NotImplementedError

    x_train, y_train = create_segments_and_labels(df_train,
                                                  time_periods,
                                                  step_distance,
                                                  LABEL)

    x_test, y_test = create_segments_and_labels(df_test,
                                                time_periods,
                                                step_distance,
                                                LABEL)

    # Set input & output dimensions
    num_classes = le.classes_.size

    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')

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

    return x_train, y_train_hot, x_test, y_test_hot

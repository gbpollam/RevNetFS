import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from utils.prepare_data import *

from layers.RevLayerKeras import RevLayerKeras, SmallerRevLayerKeras
from layers.StackChannels import StackChannels

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
    df = read_data(file_path)

    # Manually plotting user and activity pie charts (df.plot.pie is too slow)
    # df.plot.pie(y='user-id')
    # df.plot.pie(y='activity')
    activity_vc = df['activity'].value_counts()
    activity_labels = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    # plt.pie(activity_vc, labels=activity_labels, autopct='%.2f')
    # plt.show()

    userid_vc = df['user-id'].value_counts().sort_index()
    userid_labels = userid_vc.index.to_list()
    # plt.pie(userid_vc, labels=userid_labels)
    # plt.show()

    # Plot processed data
    # Define column name of the label vector
    LABEL = 'ActivityEncoded'
    # Transform the labels from String to Integer via LabelEncoder
    le = preprocessing.LabelEncoder()
    # Add a new column to the existing DataFrame with the encoded values
    df[LABEL] = le.fit_transform(df['activity'].values.ravel())

    # Differentiate between test set and training set
    df_test = df[df['user-id'] > 28]
    df_train = df[df['user-id'] <= 28]

    scaler_type = 'minmax'

    if scaler_type == 'minmax':
        df_train, scaler = fit_minmax(df_train)
        df_test = minmax(df_test, scaler)
    elif scaler_type == 'gauss_rank':
        df_train, scaler = fit_gauss_rank(df_train)
        df_test = gauss_rank(df_test, scaler)
    else:
        raise NotImplementedError

    x_train, y_train, user_id_train = new_create_segments_and_labels(df_train,
                                                               time_steps=TIME_PERIODS,
                                                               step=STEP_DISTANCE,
                                                               label_name=LABELS)

    x_test, y_test, user_id_test = new_create_segments_and_labels(df_test,
                                                             time_steps=TIME_PERIODS,
                                                             step=STEP_DISTANCE,
                                                             label_name=LABELS)

    user_id_train_vc = pd.DataFrame(user_id_train).value_counts()
    plt.pie(user_id_train_vc)
    plt.show()

    df_train = pd.Series({'x-axis': x_train[:, :, 0],
                          'y-axis': x_train[:, :, 1],
                          'z-axis': x_train[:, :, 2],
                          'activity': y_train,
                          'user-id': user_id_train})

    df_test = pd.Series({'x-axis': x_test[:, :, 0],
                         'y-axis': x_test[:, :, 1],
                         'z-axis': x_test[:, :, 2],
                         'activity': y_test,
                         'user-id': user_id_test})


if __name__ == "__main__":
    main()

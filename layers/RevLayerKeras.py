import tensorflow as tf
import keras
import pandas as pd
from keras import layers

from utils.prepare_data import prepare_data


class RevLayerKeras(keras.layers.Layer):
    def __init__(self,
                 in_channels,
                 proportion=0.5
                 ):
        super(RevLayerKeras, self).__init__()
        # Set how to split the channels
        self.in_channels = in_channels
        self.channels_x1, self.channels_x2 = self.how_to_split(proportion)
        # Define F layers
        self.F_relu1 = tf.keras.layers.ReLU()
        self.F_bn1 = tf.keras.layers.BatchNormalization()
        self.F_conv1 = tf.keras.layers.Conv1D(filters=self.channels_x1,
                                           kernel_size=3,
                                           padding='same',
                                           strides=1)
        self.F_relu2 = tf.keras.layers.ReLU()
        self.F_bn2 = tf.keras.layers.BatchNormalization()
        self.F_conv2 = tf.keras.layers.Conv1D(filters=self.channels_x2,
                                              kernel_size=3,
                                              padding='same',
                                              strides=1)
        # Define G layers
        self.G_relu1 = tf.keras.layers.ReLU()
        self.G_bn1 = tf.keras.layers.BatchNormalization()
        self.G_conv1 = tf.keras.layers.Conv1D(filters=self.channels_x2,
                                              kernel_size=3,
                                              padding='same',
                                              strides=1)
        self.G_relu2 = tf.keras.layers.ReLU()
        self.G_bn2 = tf.keras.layers.BatchNormalization()
        self.G_conv2 = tf.keras.layers.Conv1D(filters=self.channels_x1,
                                              kernel_size=3,
                                              padding='same',
                                              strides=1)

    def call(self, x, **kwargs):
        x1, x2 = tf.split(x, num_or_size_splits=(self.channels_x1, self.channels_x2), axis=2)
        F_x1 = self.F_conv2(self.F_relu2(self.F_bn2(self.F_conv1(self.F_relu1(self.F_bn1(x2))))))
        y1 = tf.add(x1, F_x1)
        y2 = x2 + self.G_conv2(self.G_relu2(self.G_bn2(self.G_conv1(self.G_relu1(self.G_bn1(y1))))))
        return tf.concat((y1, y2), axis=2)

    def how_to_split(self, proportion):
        # TODO: Add a check to see that the number of channels are integer
        channels_x1 = self.in_channels*proportion
        channels_x2 = self.in_channels - channels_x1
        return int(channels_x1), int(channels_x2)

    def split(self, x, n_filter):
        x1 = x[:, :, :n_filter // 2]
        x2 = x[:, :, n_filter // 2:]
        return x1, x2

    def concat(self, *argv):
        return tf.concat(list(argv), axis=2)

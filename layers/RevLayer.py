import math

import tensorflow as tf
import numpy as np

import time

from layers.Layer import Layer


class RevLayer(Layer):
    def __init__(self, input_shape,
                 proportion=.5
                 ):
        super().__init__()
        self.in_channels = input_shape[1]
        self.channels_x1, self.channels_x2 = self.how_to_split(proportion)

    # Utility functions
    def how_to_split(self, proportion):
        # TODO: Add a check to see that the number of channels are integer
        channels_x1 = self.in_channels*proportion
        channels_x2 = self.in_channels - channels_x1
        return int(channels_x1), int(channels_x2)

    def split(self, input, channels_x1, channels_x2):
        return tf.split(input, num_or_size_splits=(channels_x1, channels_x2), axis=1)


    def forward(self, input):
        x1, x2 = self.split(input, self.channels_x1, self.channels_x2)

    def backward_ni(self, a_gradient, learning_rate):
        pass

    def batch_backward_ni(self, a_gradient, learning_rate, batch_size):
        pass


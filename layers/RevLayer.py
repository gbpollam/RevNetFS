import math

import tensorflow as tf
import numpy as np

import time

from layers.Layer import Layer


class RevLayer(Layer):
    def __init__(self, input_shape, proportion=.5):
        super().__init__()
        self.in_channels = input_shape[1]
        self.channels_x1, self.channels_x2 = self.how_to_split(proportion)

    # Utility functions
    def how_to_split(self, proportion):
        pass

    def forward(self, input):
        pass

    def backward_ni(self, a_gradient, learning_rate):
        pass

    def batch_backward_ni(self, a_gradient, learning_rate, batch_size):
        pass


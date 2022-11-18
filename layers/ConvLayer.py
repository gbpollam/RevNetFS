import math

import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class ConvLayer(Layer):
    def __init__(self,
                 input_channels,
                 kernel_size=3,
                 num_filters=16,
                 stride=1):
        super().__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride

        # Initialize weights (Xavier) and biases
        stddev = math.sqrt(2./(kernel_size*num_filters))
        self.weights = tf.random.normal([num_filters*input_channels, kernel_size, 1], mean=0, stddev=stddev)
        self.biases = tf.constant(0.01, shape=(num_filters, 1))

    def forward(self, input):
        pass


    def backward(self, input, a_gradient, learning_rate):
        pass
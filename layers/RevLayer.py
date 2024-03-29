import math

import tensorflow as tf
import numpy as np

import time

from layers.Layer import Layer
from layers.ConvLayer import ConvLayer
from layers.ActivationLayer import ActivationLayer


class RevLayer(Layer):
    def __init__(self,
                 input_shape,
                 proportion=.5
                 ):
        super().__init__()
        self.name = "RevLayer"
        # Set how to split the channels
        self.in_channels = input_shape[1]
        self.channels_x1, self.channels_x2 = self.how_to_split(proportion)
        # Define F layer
        self.F_conv = ConvLayer(input_shape=(input_shape[0], self.channels_x2),
                                kernel_size=3,
                                num_filters=self.channels_x1,
                                stride=1,
                                padding='SAME')
        self.F_relu = ActivationLayer('relu')
        # Define G layer
        self.G_conv = ConvLayer(input_shape=(input_shape[0], self.channels_x1),
                                kernel_size=3,
                                num_filters=self.channels_x2,
                                stride=1,
                                padding='SAME')
        self.G_relu = ActivationLayer('relu')

    # Utility functions:

    # This makes it so that the inputs are not saved for this layer
    def needs_inputs(self):
        return False

    def how_to_split(self, proportion):
        # TODO: Add a check to see that the number of channels are integer
        channels_x2 = self.in_channels * proportion
        channels_x1 = self.in_channels - channels_x2
        return int(channels_x1), int(channels_x2)

    def split(self, input, channels_x1, channels_x2):
        return tf.split(input, num_or_size_splits=(channels_x1, channels_x2), axis=1)

    # Forward and Backward
    def forward(self, input):
        x1, x2 = self.split(input, self.channels_x1, self.channels_x2)
        F_x2 = self.F_relu.forward(self.F_conv.forward(x2))
        y1 = tf.add(x1, F_x2)
        G_y1 = self.G_relu.forward(self.G_conv.forward(y1))
        y2 = tf.add(x2, G_y1)
        return tf.concat((y1, y2), axis=1)

    def backward_rev(self, output, a_gradient, learning_rate):
        # Recompute the activations:
        y1, y2 = self.split(output, self.channels_x1, self.channels_x2)
        z1 = y1
        x2 = y2 - self.G_relu.forward(self.G_conv.forward(z1))
        x1 = z1 - self.F_relu.forward(self.F_conv.forward(x2))
        # BACKPROPAGATION
        y_gradient1, y_gradient2 = self.split(a_gradient, self.channels_x1, self.channels_x2)
        # bw through G
        input_relu = self.G_conv.forward(z1)
        loss = self.G_relu.backward(input_relu, y_gradient2, learning_rate)
        loss = self.G_conv.backward(z1, loss, learning_rate)
        x_gradient1 = tf.add(loss, y_gradient1)
        # bw through F
        input_relu = self.F_conv.forward(x2)
        loss = self.F_relu.backward(input_relu, loss, learning_rate)
        x_gradient2 = tf.add(self.F_conv.backward(x2, loss, learning_rate), y_gradient2)
        return tf.concat((x_gradient1, x_gradient2), axis=1), tf.concat((x1, x2), axis=1)

    def batch_backward_rev(self, output, a_gradient, learning_rate, batch_size):
        raise NotImplementedError("Batch backward not yet implemented for RevLayer!")


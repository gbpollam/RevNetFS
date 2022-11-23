import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class GAPLayer(Layer):
    def __init__(self):
        super().__init__()
        self.num_channels = None

    def forward(self, input):
        self.num_channels = input.get_shape()[0]
        output = tf.math.reduce_mean(input, axis=0)
        output = tf.expand_dims(output, axis=1)
        return output

    def backward_ni(self, a_gradient, learning_rate):
        tensors_to_concat = []
        for i in range(self.num_channels):
            tensors_to_concat.append(tf.math.divide(a_gradient, self.num_channels))
        return tf.transpose(tf.concat(tensors_to_concat, axis=1))

    def needs_inputs(self):
        return False

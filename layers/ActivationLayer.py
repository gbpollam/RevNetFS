import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class ActivationLayer(Layer):
    def __init__(self,
                 function_name):
        super().__init__()
        self.name = "ActivationLayer"
        self.function_name = function_name
        if function_name != 'relu' and function_name != 'softmax':
            raise NotImplementedError

    def forward(self, input):
        if self.function_name == 'relu':
            return tf.nn.relu(input)
        elif self.function_name == 'softmax':
            return tf.nn.softmax(input, axis=0)

    def backward(self, input, a_gradient, learning_rate):
        if self.function_name == 'relu':
            a = tf.nn.relu(input)
            x_gradient = tf.math.multiply(a_gradient, tf.experimental.numpy.heaviside(a, 0.5))
            x_gradient = tf.clip_by_value(x_gradient, -10, 10)
            return x_gradient
        # This assumes that it is the last layer, and a_gradient is a one-hot encoding of the targets
        elif self.function_name == 'softmax':
            x_gradient = tf.nn.softmax(input, axis=0) - a_gradient
            x_gradient = tf.clip_by_value(x_gradient, -10, 10)
            return x_gradient



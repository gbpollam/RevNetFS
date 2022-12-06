import tensorflow as tf
import numpy as np

from layers.Layer import Layer


# TODO: Implement the version with the stride and padding, now it uses only window_size
class AvgPool1d(Layer):
    def __init__(self,
                 window_size=2,
                 stride=1,
                 padding='same'):
        super().__init__()
        self.name = "AvgPool1d"
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.target_length = None

    def forward(self, input):
        self.target_length = input.get_shape()[0]
        count = 0
        dump = []
        dump_res = []
        for i in range(self.target_length):
            count += 1
            dump.append(input[i])
            if count == self.window_size:
                if i >= (self.target_length - self.window_size):
                    for j in range(i, (self.target_length)):
                        dump.append(input[j])
                dump_res.append(tf.reduce_mean(dump, axis=0))
                dump = []
                count = 0
        return tf.stack(dump_res, axis=0)

    def backward_ni(self, a_gradient, learning_rate):
        dump = []
        for i in range(a_gradient.get_shape()[0] - 1):
            for j in range(self.window_size):
                dump.append(tf.math.divide(a_gradient[i], self.window_size))
        missing = self.target_length - len(dump)
        for k in range(missing):
            dump.append(tf.math.divide(a_gradient[a_gradient.get_shape()[0] - 1], missing))
        return tf.stack(dump, axis=0)

    def needs_inputs(self):
        return False



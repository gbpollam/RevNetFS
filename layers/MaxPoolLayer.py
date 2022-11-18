import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class ActivationLayer(Layer):
    def __init__(self,
                 window_size, stride, padding):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        return tf.nn.avg_pool1d(input, self.window_size, self.stride, self.padding)

    def backward(self):
        pass


import math

import tensorflow as tf
import numpy as np

import time

from layers.Layer import Layer

class RevLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        pass

    def backward_ni(self, a_gradient, learning_rate):
        pass

    def batch_backward_ni(self, a_gradient, learning_rate, batch_size):
        pass


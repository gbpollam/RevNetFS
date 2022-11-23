import math

import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class ConvLayer(Layer):
    def __init__(self,
                 input_shape,
                 kernel_size=3,
                 num_filters=16,
                 stride=1,
                 padding='VALID'):
        super().__init__()
        self.input_shape = input_shape
        self.input_channels = input_shape[1]
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

        # Initialize weights (Xavier)
        stddev = math.sqrt(2./(kernel_size*num_filters))
        # self.weights = tf.random.normal([num_filters*input_channels, kernel_size, 1], mean=0, stddev=stddev)
        self.weights = tf.random.normal([kernel_size, self.input_channels, num_filters], mean=0, stddev=stddev)

        # Initialize the bias (must have the same shape as the output)
        if padding == 'VALID':
            output_shape = int(tf.math.ceil((input_shape[0] - kernel_size + 1) / stride).numpy())
        elif padding == 'SAME':
            output_shape = int(tf.math.ceil(input_shape[0] / stride).numpy())
        else:
            raise NotImplementedError("Only VALID and SAME padding implemented!")
        self.biases = tf.constant(0.01, shape=(1, num_filters))

        self.w_gradient_saved = None
        self.b_gradient_saved = None
        self.batch_counter = 0

    def forward(self, input):
        input = tf.expand_dims(input, axis=0)
        # The two versions give the same result
        # output = tf.nn.conv1d(input=input, filters=self.weights, stride=self.stride, padding=self.padding)
        output = tf.nn.convolution(input=input, filters=self.weights, strides=self.stride, padding=self.padding)
        output = tf.squeeze(output)
        output += self.biases
        return output

    def gradient_descent(self, w_gradient, b_gradient, learning_rate):
        self.weights -= learning_rate * w_gradient
        self.biases -= learning_rate * b_gradient

    # TODO: Check whether it is correct even for other cases of stride
    def backward(self, input, a_gradient, learning_rate):
        b_gradient = a_gradient
        a_gradient = tf.expand_dims(a_gradient, axis=0)
        x_gradient = tf.squeeze(
            tf.nn.conv1d(input=a_gradient, filters=self.weights, stride=self.stride, padding='SAME'))
        w_gradient = tf.squeeze(tf.nn.conv1d(input=a_gradient, filters=input, stride=self.stride, padding='SAME'))

        w_gradient = tf.clip_by_value(w_gradient, -10, 10)
        b_gradient = tf.clip_by_value(b_gradient, -10, 10)
        x_gradient = tf.clip_by_value(x_gradient, -10, 10)

        self.gradient_descent(w_gradient, b_gradient, learning_rate)

        return x_gradient

    def batch_backward(self, input, a_gradient, learning_rate, batch_size):
        b_gradient = a_gradient
        a_gradient = tf.expand_dims(a_gradient, axis=0)
        paddings = ([0, 0], [2, 2], [0, 0])
        x_gradient = tf.squeeze(
            tf.nn.convolution(input=tf.pad(a_gradient, paddings, "CONSTANT"),
                              filters=tf.transpose(self.weights, [0, 2, 1]),
                              strides=self.stride,
                              padding=self.padding))
        print("x_gradient size:")
        print(tf.nn.conv1d(input=a_gradient, filters=self.weights, stride=self.stride, padding='SAME').get_shape())
        print("a_gradient size:")
        print(a_gradient.get_shape())
        print("input size:")
        print(input.get_shape())
        # w_gradient = tf.squeeze(tf.nn.conv1d(input=a_gradient, filters=input, stride=self.stride, padding='SAME'))
        w_gradient = tf.nn.conv1d_transpose(input=a_gradient, filters=input, strides=self.stride, padding='SAME')
        print("w_gradient size:")
        print(w_gradient.get_shape())

        w_gradient = tf.clip_by_value(w_gradient, -10, 10)
        b_gradient = tf.clip_by_value(b_gradient, -10, 10)
        x_gradient = tf.clip_by_value(x_gradient, -10, 10)

        if self.w_gradient_saved is None:
            self.w_gradient_saved = w_gradient
        else:
            self.w_gradient_saved = tf.math.add(w_gradient, self.w_gradient_saved)
        if self.b_gradient_saved is None:
            self.b_gradient_saved = b_gradient
        else:
            self.b_gradient_saved = tf.math.add(b_gradient, self.b_gradient_saved)

        self.batch_counter += 1

        if self.batch_counter == (batch_size - 1):
            self.batch_counter = 0
            self.gradient_descent(tf.divide(w_gradient, batch_size), tf.divide(b_gradient, batch_size), learning_rate)
            self.w_gradient_saved = None
            self.b_gradient_saved = None

        return x_gradient



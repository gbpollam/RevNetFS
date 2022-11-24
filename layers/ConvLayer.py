import math

import tensorflow as tf
import numpy as np

import time

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
            self.output_shape = int(tf.math.ceil((input_shape[0] - kernel_size + 1) / stride).numpy())
        elif padding == 'SAME':
            self.output_shape = int(tf.math.ceil(input_shape[0] / stride).numpy())
        else:
            raise NotImplementedError("Only VALID and SAME padding implemented!")
        self.biases = tf.constant(0.01, shape=(1, num_filters))
        # (output_shape, num_filters)

        self.w_gradient_saved = None
        self.b_gradient_saved = None
        self.batch_counter = 0

    def forward(self, input):
        input = tf.expand_dims(input, axis=0)
        # The two versions give the same result
        # output = tf.nn.conv1d(input=input, filters=self.weights, stride=self.stride, padding=self.padding)
        output = tf.nn.convolution(input=input, filters=self.weights, strides=self.stride, padding=self.padding)
        output = tf.squeeze(output)
        stacked_biases = tf.repeat(self.biases, repeats=[self.output_shape], axis=0)
        output += stacked_biases
        return output

    def gradient_descent(self, w_gradient, b_gradient, learning_rate):
        self.weights -= learning_rate * w_gradient
        self.biases -= learning_rate * b_gradient

    # TODO: Check whether it is correct even for other cases of stride
    def backward(self, input, a_gradient, learning_rate):
        # Compute the b_gradient
        b_gradient = tf.expand_dims(tf.reduce_sum(a_gradient, axis=0), axis=0)

        # Compute the w_gradient (for now crudely with numpy)
        my_shape = (self.kernel_size, self.input_channels, self.num_filters)
        w_gradient = np.zeros(shape=my_shape)
        a_gradient_arr = tf.make_ndarray(tf.make_tensor_proto(a_gradient))
        input_arr = tf.make_ndarray(tf.make_tensor_proto(input))
        for n in range(my_shape[0]):
            for i in range(my_shape[1]):
                for q in range(my_shape[2]):
                    sum = 0
                    for m in range(self.input_shape[0] - self.kernel_size + 1):
                        # sum += a_gradient[m, q].numpy() * input[m+n-1, i].numpy()
                        sum += a_gradient_arr[m, q] * input_arr[m + n - 1, i]
                    w_gradient[n, i, q] = sum

        w_gradient = tf.convert_to_tensor(w_gradient, dtype=float)

        # Compute the x_gradient
        a_gradient = tf.expand_dims(a_gradient, axis=0)
        paddings = ([0, 0], [2, 2], [0, 0])
        x_gradient = tf.squeeze(
            tf.nn.convolution(input=tf.pad(a_gradient, paddings, "CONSTANT"),
                              filters=tf.transpose(self.weights, [0, 2, 1]),
                              strides=self.stride,
                              padding=self.padding))

        w_gradient = tf.clip_by_value(w_gradient, -10, 10)
        b_gradient = tf.clip_by_value(b_gradient, -10, 10)
        x_gradient = tf.clip_by_value(x_gradient, -10, 10)

        self.gradient_descent(w_gradient, b_gradient, learning_rate)

        return x_gradient

    def batch_backward(self, input, a_gradient, learning_rate, batch_size):
        # Compute the b_gradient
        b_gradient = tf.expand_dims(tf.reduce_sum(a_gradient, axis=0), axis=0)

        # Compute the w_gradient (for now crudely with numpy)
        my_shape = (self.kernel_size, self.input_channels, self.num_filters)
        w_gradient = np.zeros(shape=my_shape)
        a_gradient_arr = tf.make_ndarray(tf.make_tensor_proto(a_gradient))
        input_arr = tf.make_ndarray(tf.make_tensor_proto(input))
        for n in range(my_shape[0]):
            for i in range(my_shape[1]):
                for q in range(my_shape[2]):
                    sum = 0
                    for m in range(self.input_shape[0] - self.kernel_size + 1):
                        # sum += a_gradient[m, q].numpy() * input[m+n-1, i].numpy()
                        sum += a_gradient_arr[m, q] * input_arr[m + n - 1, i]
                    w_gradient[n, i, q] = sum

        w_gradient = tf.convert_to_tensor(w_gradient, dtype=float)

        #Compute the x_gradient
        a_gradient = tf.expand_dims(a_gradient, axis=0)
        paddings = ([0, 0], [2, 2], [0, 0])
        x_gradient = tf.squeeze(
            tf.nn.convolution(input=tf.pad(a_gradient, paddings, "CONSTANT"),
                              filters=tf.transpose(self.weights, [0, 2, 1]),
                              strides=self.stride,
                              padding=self.padding))

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



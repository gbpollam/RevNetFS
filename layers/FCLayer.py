import tensorflow as tf
import numpy as np

from layers.Layer import Layer


class FCLayer(Layer):
    def __init__(self,
                 units=32,
                 input_dim=32,
                 initialization='xavier'):
        super().__init__()
        if initialization == 'xavier':
            stddev = np.sqrt(2. / (units + input_dim))
            self.W = tf.random.normal([input_dim, units], mean=0, stddev=stddev)
        else:
            raise NotImplementedError(initialization, " initialization not implemented")
        self.b = tf.zeros([units, 1])

    def forward(self, input):
        return tf.matmul(self.W, input, transpose_a=True) + self.b

    def gradient_descent(self, w_gradient, b_gradient, learning_rate):
        self.W -= learning_rate * w_gradient
        self.b -= learning_rate * b_gradient

    def backward(self, input, a_gradient, learning_rate):
        w_gradient = tf.matmul(input, a_gradient, transpose_b=True)
        b_gradient = a_gradient
        x_gradient = tf.matmul(self.W, a_gradient)

        w_gradient = tf.clip_by_value(w_gradient, -10, 10)
        b_gradient = tf.clip_by_value(b_gradient, -10, 10)
        x_gradient = tf.clip_by_value(x_gradient, -10, 10)

        self.gradient_descent(w_gradient, b_gradient, learning_rate)

        return x_gradient

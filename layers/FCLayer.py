import tensorflow as tf
import numpy as np

from layers.Layer import Layer

# Temporary global variable, activate when comparing custom and Keras gradients
SAVE_GRADS = False

class FCLayer(Layer):
    def __init__(self,
                 units=32,
                 input_dim=32,
                 initialization='xavier'):
        super().__init__()
        self.name = "FCLayer"
        if initialization == 'xavier':
            stddev = np.sqrt(2. / (units + input_dim))
            self.W = tf.random.normal([input_dim, units], mean=0, stddev=stddev)
        else:
            raise NotImplementedError(initialization, " initialization not implemented")
        self.b = tf.zeros([units, 1])
        self.w_gradient_saved = None
        self.b_gradient_saved = None
        self.batch_counter = 0

    def forward(self, input):
        return tf.matmul(self.W, input, transpose_a=True) + self.b

    def gradient_descent(self, w_gradient, b_gradient, learning_rate):
        self.W -= learning_rate * w_gradient
        self.b -= learning_rate * b_gradient

    def backward(self, input, a_gradient, learning_rate):
        w_gradient = tf.matmul(input, a_gradient, transpose_b=True)
        b_gradient = a_gradient
        x_gradient = tf.matmul(self.W, a_gradient)

        # Save tensors to a string
        if SAVE_GRADS:
            np.save('../results/FC_w_gradient_custom.npy', w_gradient.numpy())
            np.save('../results/FC_b_gradient_custom.npy', b_gradient.numpy())


        w_gradient = tf.clip_by_value(w_gradient, -10, 10)
        b_gradient = tf.clip_by_value(b_gradient, -10, 10)
        x_gradient = tf.clip_by_value(x_gradient, -10, 10)

        self.gradient_descent(w_gradient, b_gradient, learning_rate)

        return x_gradient

    def batch_backward(self, input, a_gradient, learning_rate, batch_size):
        w_gradient = tf.matmul(input, a_gradient, transpose_b=True)
        b_gradient = a_gradient
        x_gradient = tf.matmul(self.W, a_gradient)

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

        if self.batch_counter == (batch_size-1):
            self.batch_counter = 0
            self.gradient_descent(tf.divide(w_gradient, batch_size), tf.divide(b_gradient, batch_size), learning_rate)
            self.w_gradient_saved = None
            self.b_gradient_saved = None

        return x_gradient

    def get_weights_biases(self):
        return self.W, self.b

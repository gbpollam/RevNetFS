import tensorflow as tf


class Layer:
    def __init__(self):
        self.id = 0
        self.batch_counter = 0

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, input, a_gradient, learning_rate):
        raise NotImplementedError

    def backward_ni(self, a_gradient, learning_rate):
        raise NotImplementedError

    # backward step when mini-batch training, default just calls normal backward
    def batch_backward(self, input, a_gradient, learning_rate, batch_size):
        return self.backward(input, a_gradient, learning_rate)

    def batch_backward_ni(self, a_gradient, learning_rate, batch_size):
        return self.backward_ni(a_gradient, learning_rate)

    # Sets the id of the layer in the network (i.e. layer's number)
    def set_id(self, id):
        self.id = id

    def needs_inputs(self):
        return True

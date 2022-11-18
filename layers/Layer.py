import tensorflow as tf


class Layer:
    def __init__(self):
        self.id = 0

    # computes the output Y of a layer for a given input X
    def forward(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward(self, input, a_gradient, learning_rate):
        raise NotImplementedError

    def backward_ni(self, a_gradient, learning_rate):
        raise NotImplementedError

    # Sets the id of the layer in the network (i.e. layer's number)
    def set_id(self, id):
        self.id = id

    def needs_inputs(self):
        return True

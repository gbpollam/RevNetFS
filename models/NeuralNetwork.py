import tensorflow as tf
import numpy as np
from tqdm import tqdm
import math


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.inputs = {}

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)
        layer.set_id(len(self.layers) + 1)

    # set loss to use
    def set_loss(self, loss):
        self.loss = loss

    # predict output for given input
    def predict(self, input_data):
        outputs = []
        for i in range(input_data.get_shape()[0]):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            outputs.append(output)
        return outputs

    def sgd(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            x_train = tf.random.shuffle(x_train)
            tot_loss = 0
            # tqdm(range(x_train.get_shape()[0]), desc=("Training epoch ", epoch, " of ", epochs))
            print("Training epoch ", (epoch+1), " of ", epochs, ":")
            for i in tqdm(range(x_train.get_shape()[0])):
                # Forward Pass
                self.inputs = {}
                output = x_train[i]
                target = y_train[i]
                for layer in self.layers:
                    if layer.needs_inputs():
                        self.inputs[layer.id] = output
                    output = layer.forward(output)

                # Backward Pass
                loss = self.loss(target, output).numpy()

                tot_loss += loss

                if math.isnan(tot_loss):
                    print("Iteration: ", i)
                    raise AssertionError

                # TODO: This works only with softmax as last layer, make it general
                last = True
                for layer in reversed(self.layers):
                    print(layer.id)
                    if last:
                        loss = layer.backward(self.inputs[layer.id], target, learning_rate)
                    elif layer.needs_inputs():
                        loss = layer.backward(self.inputs[layer.id], loss, learning_rate)
                    else:
                        loss = layer.backward_ni(loss, learning_rate)
                    last = False
            print("Epoch Loss: ", tot_loss)

    def minibatch_gd(self, x_train, y_train, batch_size, epochs, learning_rate):
        for epoch in range(epochs):
            x_train = tf.random.shuffle(x_train)
            tot_loss = 0
            # tqdm(range(x_train.get_shape()[0]), desc=("Training epoch ", epoch, " of ", epochs))
            print("Training epoch ", (epoch+1), " of ", epochs, ":")
            for i in tqdm(range(x_train.get_shape()[0])):
                # Forward Pass
                self.inputs = {}
                output = x_train[i]
                target = y_train[i]
                for layer in self.layers:
                    if layer.needs_inputs():
                        self.inputs[layer.id] = output
                    output = layer.forward(output)

                # Backward Pass
                loss = self.loss(target, output).numpy()

                tot_loss += loss

                if math.isnan(tot_loss):
                    print("Iteration: ", i)
                    raise AssertionError

                # TODO: This works only with softmax as last layer, make it general
                last = True
                for layer in reversed(self.layers):
                    if last:
                        loss = layer.batch_backward(self.inputs[layer.id], target, learning_rate, batch_size)
                    elif layer.needs_inputs():
                        loss = layer.batch_backward(self.inputs[layer.id], loss, learning_rate, batch_size)
                    else:
                        loss = layer.batch_backward_ni(loss, learning_rate, batch_size)
                    last = False
            print("Epoch Loss: ", tot_loss)

    # WIP
    def split_val(self, x_train, y_train, validation_split):
        tot_data = tf.shape(x_train)[0]
        val_data = int(tot_data * validation_split)

        x_val = tf.zeros([val_data, tf.shape(x_train)[1], tf.shape(x_train)[2]])
        y_val = tf.zeros([val_data, tf.shape(y_train)[1]])
        for i in range(val_data):
            datum = np.random.uniform(0, tot_data-i)
            x_val[i] = x_train[datum]
            y_val[i] = y_train[datum]
            x_train = tf.concat([x_train[0:datum], x_train[datum+1:]], axis=0)
            y_train = tf.concat([y_train[0:datum], y_train[datum + 1:]], axis=0)
        return x_train, y_train, x_val, y_val

    # train the network
    def fit(self, x_train, y_train, batch_size=1, epochs=10, learning_rate=0.1, validation_split=0.2):
        if batch_size == 1:
            return self.sgd(x_train, y_train, epochs, learning_rate)
        else:
            return self.minibatch_gd(x_train, y_train, batch_size, epochs, learning_rate)

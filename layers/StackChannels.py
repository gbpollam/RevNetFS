import tensorflow as tf
import keras
from keras import layers


class StackChannels(keras.layers.Layer):
    def __init__(self,
                 copies=1
                 ):
        super(StackChannels, self).__init__()
        self.copies = copies

    def call(self, x, **kwargs):
        return tf.tile(x, multiples=[1, 1, self.copies])

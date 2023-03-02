import tensorflow as tf
import keras
from keras import layers
from keras.layers import Reshape


class InvertibleDownsampling(keras.layers.Layer):
    def __init__(self,
                 target_shape,
                 paddings,
                 bool_pad=False):
        super(InvertibleDownsampling, self).__init__()
        self.paddings = paddings
        self.bool_pad = bool_pad
        self.reshape = Reshape(target_shape=target_shape)

    def call(self, x, **kwargs):
        if self.bool_pad:
            x = tf.pad(x, paddings=self.paddings)
        idx1 = []
        idx2 = []
        spatial_dim = int(x.shape[1])
        for i in range(0, spatial_dim, step=2):
            idx1.append(i)
            idx2.append(i+1)
        idx = idx1 + idx2
        x = tf.gather(x, indices=idx, axis=1)
        x = self.reshape(x)
        # x = tf.reshape(x, tf.TensorShape([self.batch_size, int(x.shape[1]/2),
        #                   int(x.shape[2]*2)]))
        return x

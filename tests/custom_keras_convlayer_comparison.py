import tensorflow as tf
import keras
from keras import layers
from keras import backend as k
import pandas as pd
import numpy as np

from utils.prepare_data import prepare_data

from models.NeuralNetwork import NeuralNetwork
from layers.FCLayer import FCLayer
from layers.ActivationLayer import ActivationLayer
from layers.AvgPool1d import AvgPool1d
from layers.ConvLayer import ConvLayer
from layers.GAPLayer import GAPLayer
from layers.RevLayer import RevLayer
from layers.MutantRevLayer import MutantRevLayer

# Set some standard parameters upfront
pd.options.display.float_format = '{:.1f}'.format


# DEFINE GLOBAL VARIABLES
# Same labels will be reused throughout the program
LABELS = ['Downstairs',
          'Jogging',
          'Sitting',
          'Standing',
          'Upstairs',
          'Walking']
# The number of steps within one time segment
TIME_PERIODS = 20
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 10

PADDING = 'VALID'


def get_gradients(model: tf.keras.Model, x, y_true):
    """Return the gradient of every trainable weight in model"""
    with tf.GradientTape() as tape:
        loss = model.compiled_loss(y_true, model(x))
        # loss_fun = tf.keras.losses.BinaryCrossentropy()
        # loss = loss_fun(y_true, model(x))

    print("Loss computed by Keras: ", loss)

    return tape.gradient(loss, model.trainable_weights)


def main():
    x_train = tf.random.normal([20, 3], mean=0, stddev=5)
    y_train = tf.constant(shape=(1, 6), value=[0, 0, 0, 0, 0, 1], dtype=float)
    y_train_t = tf.constant(shape=(6, 1), value=[0, 0, 0, 0, 0, 1], dtype=float)

    if PADDING == 'VALID':
        model_custom = NeuralNetwork()
        model_custom.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=6, stride=1))
        model_custom.add(GAPLayer())
        model_custom.add(ActivationLayer('softmax'))

        model_custom.set_loss(tf.keras.losses.BinaryCrossentropy())

        w_conv, b_conv = model_custom.layers[0].get_weigths_biases()
        wb_conv = [tf.make_ndarray(tf.make_tensor_proto(w_conv)),
                   tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(b_conv)))]

        class ModelKeras(tf.keras.Model):
            def __init__(self):
                super(ModelKeras, self).__init__()
                self.layer1 = keras.layers.Conv1D(filters=6, kernel_size=3)
                self.layer2 = keras.layers.GlobalAvgPool1D()
                self.classifier = keras.layers.Activation('softmax')

            def call(self, inputs):
                target = tf.constant(shape=(1, 6), value=[0, 0, 0, 0, 0, 1], dtype=tf.int32)
                self_loss = tf.keras.losses.CategoricalCrossentropy()
                with tf.GradientTape(persistent=True) as tape:
                    x1 = self.layer1(inputs)
                    x2 = self.layer2(x1)
                    output = self.classifier(x2)
                    loss = self_loss(target, output)
                gradients = [tape.gradient(loss, self.layer1.trainable_weights)]
                a_gradients = [tape.gradient(loss, inputs), tape.gradient(loss, x1), tape.gradient(loss, x2)]
                return output, gradients, a_gradients, loss

        # model_keras = keras.Model(input_keras, output_keras)
        model_keras = ModelKeras()
        model_keras.build(input_shape=(1, 20, 3))

        I = tf.keras.Input(shape=(20, 3))
        model_keras(I)

        model_keras.layers[0].set_weights(wb_conv)

        output_keras, gradients_keras, a_gradients_keras, loss = model_keras.call(tf.expand_dims(x_train, axis=0))

        gradients_custom, output_custom, target_custom, loss_custom, loss_tf_custom = \
            model_custom.get_gradients_for_datum(x_train, y_train_t, 0.01)

        print("------------------------------------------Comparing the losses----------"
              "---------------------------------")
        print("Keras loss: ", loss)
        print("Custom loss: ", loss_tf_custom)
        print("Difference: ", loss - loss_tf_custom)

        print("------------------------------------------Comparing the outputs----------"
              "---------------------------------")
        print("Keras output: ", output_keras)
        print("Custom loss: ", output_custom)
        print("Difference: ", output_keras - output_custom)

        print("------------------------------------------Comparing the a_gradients----------"
              "---------------------------------")
        print("Keras a_gradients: ", a_gradients_keras)
        print("Custom a_gradients: ", gradients_custom)
        print("Differences:", a_gradients_keras[2] - tf.transpose(gradients_custom[0]),
              tf.squeeze(a_gradients_keras[1]) - gradients_custom[1])

        print("------------------------------------------Comparing the gradients of weights and biases----------"
              "---------------------------------")
        print(gradients_keras)
        print("Difference between Conv_w_gradients:")
        print(gradients_keras[0][0].numpy() - np.load('../results/Conv_w_gradient_custom.npy'))

        print("Difference between Conv_b_gradients:")
        print(gradients_keras[0][1].numpy() - np.load('../results/Conv_b_gradient_custom.npy'))
        return 0

    elif PADDING == 'SAME':
        model_custom = NeuralNetwork()
        model_custom.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=6, stride=1, padding='SAME'))
        model_custom.add(GAPLayer())
        model_custom.add(ActivationLayer('softmax'))

        model_custom.set_loss(tf.keras.losses.BinaryCrossentropy())

        w_conv, b_conv = model_custom.layers[0].get_weigths_biases()
        wb_conv = [tf.make_ndarray(tf.make_tensor_proto(w_conv)),
                   tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(b_conv)))]

        class ModelKeras(tf.keras.Model):
            def __init__(self):
                super(ModelKeras, self).__init__()
                self.layer1 = keras.layers.Conv1D(filters=6, kernel_size=3, padding='same')
                self.layer2 = keras.layers.GlobalAvgPool1D()
                self.classifier = keras.layers.Activation('softmax')

            def call(self, inputs):
                target = tf.constant(shape=(1, 6), value=[0, 0, 0, 0, 0, 1], dtype=tf.int32)
                self_loss = tf.keras.losses.CategoricalCrossentropy()
                with tf.GradientTape(persistent=True) as tape:
                    x1 = self.layer1(inputs)
                    x2 = self.layer2(x1)
                    output = self.classifier(x2)
                    loss = self_loss(target, output)
                gradients = [tape.gradient(loss, self.layer1.trainable_weights)]
                a_gradients = [tape.gradient(loss, inputs), tape.gradient(loss, x1), tape.gradient(loss, x2)]
                return output, gradients, a_gradients, loss

        # model_keras = keras.Model(input_keras, output_keras)
        model_keras = ModelKeras()
        model_keras.build(input_shape=(1, 20, 3))

        I = tf.keras.Input(shape=(20, 3))
        model_keras(I)

        model_keras.layers[0].set_weights(wb_conv)

        output_keras, gradients_keras, a_gradients_keras, loss = model_keras.call(tf.expand_dims(x_train, axis=0))

        gradients_custom, output_custom, target_custom, loss_custom, loss_tf_custom = \
            model_custom.get_gradients_for_datum(x_train, y_train_t, 0.01)

        print("------------------------------------------Comparing the losses----------"
              "---------------------------------")
        print("Keras loss: ", loss)
        print("Custom loss: ", loss_tf_custom)
        print("Difference: ", loss - loss_tf_custom)

        print("------------------------------------------Comparing the outputs----------"
              "---------------------------------")
        print("Keras output: ", output_keras)
        print("Custom loss: ", output_custom)
        print("Difference: ", output_keras - output_custom)

        print("------------------------------------------Comparing the a_gradients----------"
              "---------------------------------")
        print("Keras a_gradients: ", a_gradients_keras)
        print("Custom a_gradients: ", gradients_custom)
        print("Differences:", a_gradients_keras[2] - tf.transpose(gradients_custom[0]),
              tf.squeeze(a_gradients_keras[1]) - gradients_custom[1])

        print("------------------------------------------Comparing the gradients of weights and biases----------"
              "---------------------------------")
        print(gradients_keras)
        print("Difference between Conv_w_gradients:")
        print(gradients_keras[0][0].numpy() - np.load('../results/Conv_w_gradient_custom.npy'))

        print("Difference between Conv_b_gradients:")
        print(gradients_keras[0][1].numpy() - np.load('../results/Conv_b_gradient_custom.npy'))
        return 0

    # Declare a custom Conv layer
    custom_conv = ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=16, stride=1)

    weights, biases = custom_conv.get_weigths_biases()
    wb_list = [tf.make_ndarray(tf.make_tensor_proto(weights)), tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(biases)))]
    # Declare a keras model containing only a conv layer
    model_conv = keras.Sequential()
    model_conv.add(keras.Input(shape=(20, 3)))
    model_conv.add(keras.layers.Conv1D(filters=16, kernel_size=3))
    model_conv.layers[0].set_weights(wb_list)

    # Test the forward call
    fw_custom = custom_conv.forward(x_train)
    # fw_keras = model_conv.predict(x_train)
    fw_keras = model_conv.call(tf.expand_dims(x_train, axis=0))
    # print(fw_custom - fw_keras)  # Outputs all zeros so the forward is correct

    # Testing multiple layers
    model_custom = NeuralNetwork()
    model_custom.add(ConvLayer(input_shape=(20, 3), kernel_size=3, num_filters=16, stride=1))
    model_custom.add(ActivationLayer('relu'))
    model_custom.add(GAPLayer())
    model_custom.add(FCLayer(units=6, input_dim=16))
    model_custom.add(ActivationLayer('softmax'))

    model_custom.set_loss(tf.keras.losses.BinaryCrossentropy())

    w_conv, b_conv = model_custom.layers[0].get_weigths_biases()
    wb_conv = [tf.make_ndarray(tf.make_tensor_proto(w_conv)),
               tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(b_conv)))]
    w_fc, b_fc = model_custom.layers[3].get_weights_biases()
    wb_fc = [tf.make_ndarray(tf.make_tensor_proto(w_fc)),
               tf.make_ndarray(tf.make_tensor_proto(tf.squeeze(b_fc)))]

    """
    model_keras = keras.Sequential()
    model_keras.add(keras.Input(shape=(20, 3)))
    model_keras.add(keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'))
    model_keras.add(keras.layers.GlobalAvgPool1D())
    model_keras.add(keras.layers.Dense(units=6, activation='softmax'))
    """
    """
    input_keras = keras.Input(shape=(20, 3))
    x = keras.layers.Conv1D(filters=16, kernel_size=3)(input_keras)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAvgPool1D()(x)
    output_keras = keras.layers.Dense(units=6, activation='softmax')(x)
    """

    class ModelKeras(tf.keras.Model):
        def __init__(self):
            super(ModelKeras, self).__init__()
            self.layer1 = keras.layers.Conv1D(filters=16, kernel_size=3)
            self.layer2 = keras.layers.Activation('relu')
            self.layer3 = keras.layers.GlobalAvgPool1D()
            self.classifier = keras.layers.Dense(units=6, activation='softmax')

        def call(self, inputs):
            target = tf.constant(shape=(1, 6), value=[0, 0, 0, 0, 0, 1], dtype=tf.int32)
            self_loss = tf.keras.losses.CategoricalCrossentropy()
            with tf.GradientTape(persistent=True) as tape:
                x1 = self.layer1(inputs)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                output = self.classifier(x3)
                loss = self_loss(target, output)
            gradients = [tape.gradient(loss, self.layer1.trainable_weights),
                         # tape.gradient(loss, self.layer2.trainable_weights),
                         # tape.gradient(loss, self.layer3.trainable_weights),
                         tape.gradient(loss, self.classifier.trainable_weights)]
            a_gradients = [tape.gradient(loss, inputs), tape.gradient(loss, x1), tape.gradient(loss, x2),
                           tape.gradient(loss, x3)]
            return output, gradients, a_gradients, loss



    # model_keras = keras.Model(input_keras, output_keras)
    model_keras = ModelKeras()
    model_keras.build(input_shape=(1, 20, 3))

    I = tf.keras.Input(shape=(20, 3))
    model_keras(I)

    # model_keras.layers[1].set_weights(wb_conv)
    # model_keras.layers[4].set_weights(wb_fc)
    model_keras.layers[0].set_weights(wb_conv)
    model_keras.layers[3].set_weights(wb_fc)

    a, b, c, loss = model_keras.call(tf.expand_dims(x_train, axis=0))

    print("Keras loss: ", loss)
    print("Keras output: ", a)
    print("Keras gradients: ", b)
    print("Keras gradients length: ", len(b))
    print("Keras a_gradients: ", c)
    print("Keras a_gradients: ", len(c))

    print("------------------------------------------Printing grad-by-grad-------------------------------------------")

    model_keras.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
    """
    #with tf.GradientTape(persistent=True) as tape:
        loss = model_keras.compiled_loss(y_train, model_keras(tf.expand_dims(x_train, axis=0)))
        # loss_fun = tf.keras.losses.BinaryCrossentropy()
        # loss = loss_fun(y_true, model(x))
    

    loss_fun = tf.keras.losses.CategoricalCrossentropy()
    exp_loss = loss_fun(y_train, model_keras(tf.expand_dims(x_train, axis=0)))
    print("model_keras loss computed explicitly: ", exp_loss)
    print("Loss computed by Keras: ", loss)
    print("Keras output: ", model_keras(tf.expand_dims(x_train, axis=0)))
    print("Keras target: ", y_train)

    gradients_keras = tape.gradient(loss, model_keras.trainable_weights)
    print("Layer 3 inputs: ", model_keras.layers[2].input)
    a_gradients_keras3 = tape.gradient(loss, model_keras.layers[2].output)
    print("a_gradients_keras3: ", a_gradients_keras3)
    """
    # Test the forward call
    fw_custom = model_custom.predict_datum(x_train)
    # fw_keras = model_keras(tf.expand_dims(x_train, axis=0))
    # fw_keras = model_keras(tf.expand_dims(x_train, axis=0))
    # fw_keras = model_keras.call(tf.expand_dims(x_train, axis=0))
    print("Comparison of the networks' outputs:")
    print(fw_custom - tf.transpose(a, [1, 0]))  # Outputs all zeros so the forward is correct

    # model_keras.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.optimizers.SGD(learning_rate=0.01))

    # gradients_keras = get_gradients(model_keras, tf.expand_dims(x_train, axis=0), y_train)
    gradients_custom = model_custom.get_gradients_for_datum(x_train, y_train_t, 0.01)

    print("--------------------------------------Keras gradients------------------------------------------------")
    """
    print(gradients_keras)
    print("gradients_keras size: ", len(gradients_keras))
    it = 0
    for grad in gradients_keras:
        print("Size of keras gradient i: ", grad.get_shape())

    np.save('../results/Conv_w_gradient_keras.npy', gradients_keras[0].numpy())
    np.save('../results/Conv_b_gradient_keras.npy', tf.expand_dims(gradients_keras[1], axis=0).numpy())
    np.save('../results/FC_w_gradient_keras.npy', gradients_keras[2].numpy())
    np.save('../results/FC_b_gradient_keras.npy', tf.expand_dims(gradients_keras[3], axis=1).numpy())
    """

    print("--------------------------------------Custom implementation gradients-----------------------"
          "-------------------------")
    print(gradients_custom)

    print("--------------------------------------Difference between keras and custom parameter gradients--"
          "----------------------------------------------")
    print("Difference between Conv_w_gradients:")
    print(b[0][0].numpy() - np.load('../results/Conv_w_gradient_custom.npy'))

    print("Difference between Conv_b_gradients:")
    print(b[0][1].numpy() - np.load('../results/Conv_b_gradient_custom.npy'))

    print("Difference between FC_w_gradients:")
    print(b[1][0].numpy() - np.load('../results/FC_w_gradient_custom.npy'))

    print("Difference between FC_b_gradients:")
    print(b[1][1].numpy() - np.load('../results/FC_b_gradient_custom.npy'))

    """
    raise InterruptedError
    # Test the backward call
    # Get the gradients of the keras layer
    x_tensor = tf.expand_dims(x_train, axis=0)

    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = model_conv.call(x_tensor)

    result = output
    gradients = t.gradient(output, x_tensor)

    # get the gradient of the custom layer
    x_gradient, w_gradient, b_gradient = custom_conv.compute_gradients(x_train, fw_custom)

    print(x_gradient - gradients)
    """



if __name__ == "__main__":
    main()

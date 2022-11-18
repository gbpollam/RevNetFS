import tensorflow as tf
import numpy as np

'''
W = tf.random.normal([5, 4], mean=0, stddev=5)
b = tf.random.normal([4,1], mean=0, stddev=5)
input = tf.random.normal([5,1], mean=0, stddev=5)

print(W)
print(b)
print(input)

out = tf.matmul(W, input, transpose_a=True) + b

print(tf.shape(out))
print(tf.shape(out)[1])
print(out)
for i in range(tf.shape(out)[0]):
    print("hi")
'''

'''
        print("Size of input for FCLayer ", self.id, ":")
        print(input.get_shape())
        print("Size of a_gradient for FCLayer ", self.id, ":")
        print(a_gradient.get_shape())
        
        print("Size of w_gradient for FCLayer ", self.id, ":")
        print(w_gradient.get_shape())
        print("Size of b_gradient for FCLayer ", self.id, ":")
        print(b_gradient.get_shape())
        print("Size of x_gradient for FCLayer ", self.id, ":")
        print(x_gradient.get_shape())
        print("--------------------------------------------------------------------")
        
        print("Size of input for ActivationLayer ", self.id, " with function ", self.function_name, ":")
        print(input.get_shape())
        print("Size of a_gradient for ActivationLayer ", self.id, " with function ", self.function_name, ":")
        print(a_gradient.get_shape())
        
        print("Size of x_gradient for ActivationLayer ", self.id, " with function ", self.function_name, ":")
        print(x_gradient.get_shape())
        print("--------------------------------------------------------------------")
        
        print("Size of x_gradient for ActivationLayer ", self.id, " with function ", self.function_name, ":")
        print(x_gradient.get_shape())
        print("--------------------------------------------------------------------")                
'''
'''
x_train = tf.random.normal([20, 10, 5], mean=0, stddev=5)
y_train = tf.random.normal([20, 10], mean=0, stddev=5)


def split_val(x_train, y_train, validation_split):
    tot_data = int(x_train.get_shape()[0])
    val_data = int(tot_data * validation_split)
    print(val_data)

    x_val = tf.zeros([val_data, tf.shape(x_train)[1], tf.shape(x_train)[2]])
    y_val = tf.zeros([val_data, tf.shape(y_train)[1]])
    for i in range(val_data):
        datum = np.random.randint(0, tot_data - i)
        x_val[i] = x_train[datum]
        y_val[i] = y_train[datum]
        x_train = tf.concat([x_train[0:datum], x_train[datum + 1:]], axis=0)
        y_train = tf.concat([y_train[0:datum], y_train[datum + 1:]], axis=0)
    return x_val, y_val


x_val, y_val = split_val(x_train, y_train, 0.2)

print(x_train)
print(y_train)
print(x_val)
print(y_val)'''

output = tf.constant(
[[9.49871182e-01],
 [1.00511024e-04],
 [3.41062841e-04],
 [1.25845708e-02],
 [1.04204802e-04],
 [3.69983912e-02]])

target = tf.constant(
[[0.],
 [0.],
 [0.],
 [0.],
 [0.],
 [1.]])

output1 = tf.expand_dims(output, axis=0)
output2 = tf.expand_dims(output, axis=0)
output3 = tf.expand_dims(output, axis=0)

final_output = tf.concat([output1,output2,output3], axis=0)

print(final_output)


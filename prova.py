import tensorflow as tf
import numpy as np
import math

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

'''
input_channels = 3
kernel_size=3
num_filters=4
stride=1

input = tf.random.normal([20,3], mean=0, stddev=2)
weights = tf.random.normal([num_filters*input_channels, kernel_size, 1], mean=0, stddev=stddev)

'''

'''
input = tf.random.normal([20, 4], mean=0, stddev=5)

target_length = input.get_shape()[0]

window_size = 2
count = 0
dump = []
dump_res = []
for i in range(input.get_shape()[0]):
    count += 1
    dump.append(input[i])
    if count == window_size:
        if i >= (input.get_shape()[0] - window_size):
            for j in range(i, (input.get_shape()[0])):
                dump.append(input[j])
        dump_res.append(tf.reduce_mean(dump, axis=0))
        dump = []
        count = 0

# print(tf.stack(dump_res, axis=0))

# a_gradient = tf.random.normal([16, 6], mean=0, stddev=5)
a_gradient = tf.stack(dump_res, axis=0)

dump = []
for i in range(a_gradient.get_shape()[0] - 1):
    for j in range(window_size):
        dump.append(tf.math.divide(a_gradient[i], window_size))
missing = target_length - len(dump) + window_size
for k in range(target_length - len(dump)):
    dump.append(tf.math.divide(a_gradient[a_gradient.get_shape()[0] - 1], missing))
# print(tf.stack(dump, axis=0))

input_channels = 1
kernel_size = 3
num_filters = 16
stride = 1
padding = 'VALID'

stddev = math.sqrt(2./(kernel_size*num_filters))
weights = tf.random.normal([kernel_size, input_channels, num_filters], mean=0, stddev=stddev)
biases = tf.constant(0.01, shape=(input.get_shape()[0] - 2, num_filters))

input = tf.expand_dims(input, axis=0)
output = tf.nn.conv1d(input=input, filters=weights, stride=stride, padding=padding)
output = tf.squeeze(output)
output += biases
print(output)
'''

input = tf.random.normal([20, 3], mean=0, stddev=5)

kernel_size = 3
num_filters = 32
stride = 1

input_shape = (20, 3)
input_channels = input_shape[1]
padding = 'VALID'

stddev = math.sqrt(2. / (kernel_size * num_filters))
# self.weights = tf.random.normal([num_filters*input_channels, kernel_size, 1], mean=0, stddev=stddev)
weights = tf.random.normal([kernel_size, input_channels, num_filters], mean=0, stddev=stddev)

if padding == 'VALID':
    output_shape = int(tf.math.ceil((input_shape[0] - kernel_size + 1) / stride).numpy())
elif padding == 'SAME':
    output_shape = int(tf.math.ceil(input_shape[0] / stride).numpy())
else:
    raise NotImplementedError("Only VALID and SAME padding implemented!")
biases = tf.constant(0.01, shape=(output_shape, num_filters))





biases = tf.constant(0.01, shape=(1, 32))
output_shape = 7

stacked_biases = tf.repeat(biases, repeats=[output_shape], axis=0)

a_gradient = tf.random.normal([7, 64], mean=0, stddev=5)

x = tf.random.normal([10, 20, 3], mean=0, stddev=5)
y = tf.random.normal([10, 1], mean=0, stddev=5)

indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

print(x)
print(y)

x = tf.gather(x, shuffled_indices)
y = tf.gather(y, shuffled_indices)

print(x)
print(y)


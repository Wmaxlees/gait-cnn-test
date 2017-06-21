import numpy as np
import os
import tensorflow as tf

batch_size = 10

target_width = 50
input_width = 150

def alignment_layer (input_layer, alignment_width, dtype=tf.float32):
    batch_size = input_layer.get_shape().as_list()[0]
    input_length = input_layer.get_shape().as_list()[1]

    diagonal = tf.Variable(tf.linspace(0.0, 1.0, alignment_width), dtype=dtype)
    cumulative_diagonal = tf.cumsum(diagonal)

    distance = pairwise_euclidean_distance(input_layer)
    cumulative_distance = tf.cumsum(distance)
    cumulative_distance = tf.expand_dims(cumulative_distance, 2)

    cumulative_diagonal = tf.expand_dims(cumulative_diagonal, 0)
    cumulative_diagonal = tf.tile(cumulative_diagonal, [input_length, 1])
    cumulative_diagonal = tf.expand_dims(cumulative_diagonal, 0)
    cumulative_diagonal = tf.tile(cumulative_diagonal, [batch_size, 1, 1])

    difference = tf.subtract(cumulative_diagonal, cumulative_distance)
    indices = tf.argmin(tf.abs(difference), 1)

    split_batches = tf.unstack(input_layer, axis=0)
    split_indices = tf.unstack(indices, axis=0)
    result = []
    for i in range(batch_size):
        result.append(tf.gather(split_batches[i], split_indices[i]))
    result = tf.stack(result)

    return result


def pairwise_euclidean_distance (input_layer):
    original_size = input_layer.get_shape().as_list()

    subtrahend = tf.pad(input_layer, [[0, 0], [1, 0], [0, 0], [0, 0]])
    subtrahend = tf.slice(subtrahend, [0, 0, 0, 0], original_size)

    distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(input_layer, subtrahend)), axis=[2,3]))

    return distance

def fix_length (input_value, target_length):
    if input_value.shape[0] >= target_length:
        return input_value[0:target_length]

    else:
        difference = target_length - input_value.shape[0]
        for i in range(difference):
            input_value = np.concatenate((input_value, input_value[-1:]), axis=0)
        return input_value

def unsupervised_conv2d (input_layer, number_of_filters, filter_size, stride, padding='VALID', activation=None, name=None, dtype=tf.float32):
    if name is None:
        name = 'unsup_conv2d'

    # Create the basic conv3d layer
    with tf.variable_scope(name) as scope:
        channel_in = int(input_layer.get_shape()[3])
        _filters = tf.Variable(tf.random_normal(filter_size + (channel_in, number_of_filters,)), dtype=dtype)
        _biases = tf.Variable(tf.random_normal(number_of_filters))
        conv_layer = tf.nn.bias_add(tf.nn.conv2d(input_layer, _filters, (1, ) + stride + (1, ), padding), _bias)

    # Add the activation function if needed
    if activation is not None:
        _output = activation(conv_layer)
    else:
        _output = conv_layer

    output_mean, output_variance = tf.nn.moments(_output, (0, 3,))
    output_mean = tf.reduce_sum(output_mean)
    output_variance = tf.reduce_sum(output_variance)

    _, filter_variance = tf.nn.moments(_filters, [3])
    filter_variance = tf.reduce_sum(filter_variance)

    expanded_a = tf.expand_dims(_filters, 1)
    expanded_b = tf.expand_dims(_filters, 0)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b))

    _loss_contribution = 3*tf.divide(1, distances) + tf.divide(1, output_mean) + tf.sigmoid(output_variance) + tf.divide(1, filter_variance)

    return (_output, _loss_contribution, _filters)


folder = '/home/max/repos/mvsclass/data'
files = []
for label in os.listdir(folder):
    for file in os.listdir(folder + '/' + label):
        filename = folder + '/' + label + '/' + file
        files.append(filename)

with tf.Session() as sess:
    input_layer = tf.placeholder(tf.float32, shape=(batch_size, input_width, 20, 3))

    alignment = alignment_layer(input_layer, target_width)

    conv_1, conv_1_loss, filters_1 = unsupervised_conv2d(alignment, 25, (9, 3,), (1, 1,), padding='SAME')
    conv_2, conv_2_loss, filters_2 = unsupervised_conv2d(conv_1, 25, (9, 3,), (1, 1,), padding='SAME')
    residual_1 = conv_1 + conv_2
    conv_3, conv_3_loss, filters_3 = unsupervised_conv2d(residual_1, 15, (9, 3,), (1, 1,), padding='SAME')

    filters_1 = tf.transpose(filters_1, [3, 0, 1, 2])
    summary = tf.summary.FileWriter('./', sess.graph)
    tf.summary.image('filters_1', filters_1, max_outputs=25)
    summaries = tf.summary.merge_all()

    total_loss = tf.add_n((conv_1_loss, conv_2_loss, conv_3_loss))

    optimizer = tf.train.GradientDescentOptimizer(1)
    train = optimizer.minimize(total_loss)

    init = tf.global_variables_initializer()
    
    sess.run(init)


    for i in range(10000):
        choice = np.random.choice(files, size=batch_size)

        input_values = []
        for filename in choice:
            value = np.load(filename)
            value = fix_length(value, input_width)

            input_values.append(value)
        input_values = np.array(input_values)

        loss_value, _, executed_summaries = sess.run([total_loss, train, summaries], {input_layer: input_values})
        if i % 100 == 0:
            summary.add_summary(executed_summaries)
            print('Iteration %i' % (i))
            print('Loss: %f' % (loss_value))

    summary.flush()
    summary.close()

import numpy as np
import os
import tensorflow as tf

filters_1 = 25
filter_size_1 = (3, 3, )

filters_2 = 15
filter_size_2 = (4, 4, )

pool_size_1 = (2, 2, )
pool_stride_1 = (2, 2, )

pool_size_2 = (2, 2, )
pool_stride_2 = (2, 2, )

fully_connected_size_1 = 300
number_of_classes = 55

batch_size = 10

target_width = 50
input_width = 150

label_dict = {
    0: 0161134471,
    1: 1188703625,
    2: 2093270270,
    3: 2856866648,
    4: 3406791095,
    5: 3703781934,
    6: 4119085141,
    7: 5412580207,
    8: 6411969118,
    9: 6749241630,
    10: 7130873983,
    11: 7696895508,
    12: 8952080630,
    13: 9788591079,
    14: 530649692,
    16: 1505497087,
    17: 2315176643,
    18: 2972459618,
    19: 3555272131,
    20: 3850437893,
    21: 4682500780,
    22: 5540061520,
    23: 6438858438,
    24: 6841913716,
    25: 7454146520,
    26: 8094229592,
    27: 9073557956,
    28: 9844762993,
    29: 582085262,
    30: 1552923891,
    31: 2599888284,
    32: 3058548203,
    33: 3570311004,
    34: 3969346085,
    35: 4689245876,
    36: 5695924553,
    37: 6450879774,
    38: 6847315214,
    39: 7500884853,
    40: 8163847068,
    41: 9309572294,
    42: 9953639421,
    43: 943249986,
    44: 1904001353,
    45: 2832193324,
    46: 3107029332,
    47: 3658205528,
    48: 4089965496,
    49: 5317576596,
    50: 5730664247,
    51: 6497978695,
    52: 6948176890,
    53: 7534364401,
    54: 8173546750,
    55: 9703563942
}

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


folder = '/home/max/repos/mvsclass/data'
files = []
labels = []
for label in os.listdir(folder):
    labels.append(label_dict[label])
    for file in os.listdir(folder + '/' + label):
        filename = folder + '/' + label + '/' + file
        files.append(filename)


with tf.Session() as sess:
    init = tf.global_variables_initializer()

    input_layer = tf.placeholder(tf.float32, (batch_size, input_width, 20, 3))
    target_layer = tf.placeholder(tf.float32, (batch_size,))
    aligner = alignment_layer(input_layer, target_width)

    conv_1 = tf.layers.conv2d(aligner, filters_1, filter_size_1)
    pool_1 = tf.layers.max_pooling2d(conv_1, pool_size_1, pool_stride_1)

    conv_2 = tf.layers.conv2d(pool_1, filters_2, filter_size_2)
    pool_2 = tf.layers.max_pooling2d(conv_2, pool_size_2, pool_stride_2)

    flattener = tf.contrib.layers.flatten(pool_2)

    fully_connected_1 = tf.contrib.layers.fully_connected(flattener, 300)
    logits = tf.contrib.layers.fully_connected(fully_connected_1, 55)

    target = tf.one_hot(target_layer, number_of_classes)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    trainer = tf.train.AdamOptimizer()

    sess.run(init)

    for i in range(10000):
        indices = np.random.rand(batch_size)
        files = files[indices]
        targets = labels[indices]

        input_values = []
        for filename in choice:
            value = np.load(filename)
            value = fix_length(value, input_width)

            input_values.append(value)
        input_values = np.array(input_values)

        loss_value, _ = sess.run([loss, trainer], {input_layer: input_values, target_layer: targets})
        if i % 100 == 0:
            print('Iteration %i' % (i))
            print('Loss: %f' % (loss_value))

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
    161134471: 0,
    1188703625: 1,
    2093270270: 2,
    2856866648: 3,
    3406791095: 4,
    3703781934: 5,
    4119085141: 6,
    5412580207: 7,
    6411969118: 8,
    6749241630: 9,
    7130873983: 10,
    7696895508: 11,
    8952080630: 12,
    9788591079: 13,
    530649692: 14,
    1505497087: 15,
    2315176643: 16,
    2972459618: 17,
    3555272131: 18,
    3850437893: 19,
    4682500780: 20,
    5540061520: 21,
    6438858438: 22,
    6841913716: 23,
    7454146520: 24,
    8094229592: 25,
    9073557956: 26,
    9844762993: 27,
    582085262: 28,
    1552923891: 29,
    2599888284: 30,
    3058548203: 31,
    3570311004: 32,
    3969346085: 33,
    4689245876: 34,
    5695924553: 35,
    6450879774: 36,
    6847315214: 37,
    7500884853: 38,
    8163847068: 39,
    9309572294: 40,
    9953639421: 41,
    943249986: 42,
    1904001353: 43,
    2832193324: 44,
    3107029332: 45,
    3658205528: 46,
    4089965496: 47,
    5317576596: 48,
    5730664247: 49,
    6497978695: 50,
    6948176890: 51,
    7534364401: 52,
    8173546750: 53,
    9703563942: 54
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
    for file in os.listdir(folder + '/' + label):
        filename = folder + '/' + label + '/' + file
        files.append(filename)
        labels.append(label_dict[int(label)])
files = np.array(files)
labels = np.array(labels)


with tf.Session() as sess:
    input_layer = tf.placeholder(tf.float32, (batch_size, input_width, 20, 3))
    target_layer = tf.placeholder(tf.uint8, (batch_size,))
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

    trainer = tf.train.AdamOptimizer().minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10000):
        indices = np.random.randint(files.shape[0], size=batch_size)
        choices = files[indices]
        targets = labels[indices]

        input_values = []
        for filename in choices:
            value = np.load(filename)
            value = fix_length(value, input_width)

            input_values.append(value)
        input_values = np.array(input_values)

        loss_value, _ = sess.run([loss, trainer], {input_layer: input_values, target_layer: targets})
        if i % 100 == 0:
            print('Iteration %i' % (i))
            print('Loss: %f' % (loss_value))

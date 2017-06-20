import tensorflow as tf

batch_size = 25

filters_1 = 25
filter_size_1 = (7, 7, 3,)

filters_2 = 15
filter_size_2 = (4, 4, 4,)


with tf.Session() as sess:
    init = tf.global_variables_initializer()

    input_layer = tf.placeholder(tf.float32, (batch_size, 91, 20, 3))

    conv_1 = tf.layers.conv2d(input_layer, filters_1, filter_size_1)


    sess.run(init)
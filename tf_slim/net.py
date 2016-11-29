import tensorflow.contrib.slim as slim
import tensorflow as tf

# thinking
# conv
# max
# conv
# max
# fcl
# softmax

net = ''
with slim.arg_scope([slim.conv2d, slim.full_connected], padding='SAME',
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):

  for i in range(2):
    net = slim.conv2d(net, 256, [3, 3], scope='conv_' % (i+1))
    net = slim.max_pool2d(net, [2, 2], scope='pool_' % (i+1))

  net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc_1')




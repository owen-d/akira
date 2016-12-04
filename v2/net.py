import tensorflow.contrib.slim as slim
import tensorflow as tf

# thinking
# conv
# max
# conv
# max
# fcl
# softmax
def simple(images):
  # net = ''
  # with slim.arg_scope([slim.conv2d, slim.fully_connected], padding='SAME',
  #                       activation_fn=tf.nn.relu,
  #                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
  #                       weights_regularizer=slim.l2_regularizer(0.0005)):

  #   # for i in range(2):
  #   #   net = slim.conv2d(net, 256, [3, 3], scope='conv_%d' % (i+1))
  #   #   net = slim.max_pool2d(net, [2, 2], scope='pool_%d' % (i+1))

  #   for i in range(3):
  #     net = slim.conv2d(net, 256, [3, 3], scope='conv3_%d' % (i+1))

  #   net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc_1')

  #   return net
  net = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
  net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
  net = slim.layers.conv2d(net, 50, [5,5], scope='conv2')
  net = slim.layers.max_pool2d(net, [2,2], scope='pool2')
  net = slim.layers.flatten(net, scope='flatten3')
  net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
  net = slim.layers.fully_connected(net, 5, activation_fn=None, scope='fully_connected5')
  return net

# def inputs(train_dir, train, batch_size, num_epochs, one_hot_labels=False):
#   """Reads input data num_epochs times.
#   Args:
#     train: Selects between the training (True) and validation (False) data.
#     batch_size: Number of examples per returned batch.
#     num_epochs: Number of times to read the input data, or 0/None to
#     train forever.
#   Returns:
#     A tuple (images, labels), where:
#     * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
#     in the range [-0.5, 0.5].
#     * labels is an int32 tensor with shape [batch_size] with the true label,
#     a number in the range [0, mnist.NUM_CLASSES).
#     Note that an tf.train.QueueRunner is added to the graph, which
#     must be run using e.g. tf.train.start_queue_runners().
#     """
#   if not num_epochs: num_epochs = None
#   filename = os.path.join(train_dir,
#     TRAIN_FILE if train else VALIDATION_FILE)

#   with tf.name_scope('input'):
#     filename_queue = tf.train.string_input_producer(
#     [filename], num_epochs=num_epochs)

#   # Even when reading in multiple threads, share the filename
#   # queue.
#   image, label = read_and_decode(filename_queue)

#   if one_hot_labels:
#     label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

#   # Shuffle the examples and collect them into batch_size batches.
#   # (Internally uses a RandomShuffleQueue.)
#   # We run this in two threads to avoid being a bottleneck.
#   images, sparse_labels = tf.train.shuffle_batch(
#     [image, label], batch_size=batch_size, num_threads=2,
#     capacity=1000 + 3 * batch_size,
#     # Ensures a minimum amount of shuffling of examples.
#     min_after_dequeue=1000)

#   return images, sparse_labels
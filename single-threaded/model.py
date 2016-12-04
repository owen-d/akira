import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist
import preprocessing

tf.app.flags.DEFINE_integer('num_classes', 5,
  """number of classes""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
  """number of preprocess threads""")
FLAGS = tf.app.flags.FLAGS


def simple(images):
   net = slim.layers.conv2d(images, 20, [5,5], scope='conv1')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool1')
   net = slim.layers.conv2d(net, 50, [5,5], scope='conv2')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool2')
   net = slim.layers.flatten(net, scope='flatten3')
   net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
   net = slim.layers.fully_connected(net, 5, activation_fn=None, scope='fully_connected5')
   return net


# def read_and_decode(filename_queue):
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(
#         serialized_example,
#         # Defaults are not specified since both keys are required.
#         features={
#             'image/encoded': tf.FixedLenFeature([], tf.string),
#             'label': tf.FixedLenFeature([], tf.int64),
#         })

#     # Convert from a scalar string tensor (whose single string has
#     # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
#     # [mnist.IMAGE_PIXELS].
#     image = tf.decode_raw(features['image/encoded'], tf.uint8)
#     # image.set_shape([mnist.IMAGE_PIXELS])
#     image = tf.reshape(image, [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])
#     image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

#     # OPTIONAL: Could reshape into a 28x28 image and apply distortions
#     # here.  Since we are not applying any distortions in this
#     # example, and the next step expects the image to be flattened
#     # into a vector, we don't bother.

#     # Convert label from a scalar uint8 tensor to an int32 scalar.
#     label = tf.cast(features['label'], tf.int32)

#     return image, label

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
    'image/encoded': tf.FixedLenFeature([], tf.string),
    'image/class/label': tf.FixedLenFeature([], tf.int64),
    })

  image = preprocessing.preprocess_image(features['image/encoded'])
  label = features['image/class/label']
  return image, label


def inputs(train_dir, batch_size, num_epochs, one_hot_labels=False, train=True):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
    train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
    in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
    a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  TRAIN_FILE = 'train.tfrecords'
  VALIDATION_FILE = 'validation.tfrecords'

  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir,
              TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    if one_hot_labels:
      label = tf.one_hot(label, FLAGS.num_classes, dtype=tf.int32)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, num_threads=FLAGS.num_preprocess_threads,
      capacity=1000 + 3 * batch_size,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=1000)

  return images, sparse_labels
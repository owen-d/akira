import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import simple, inputs
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('train_dir', '/tmp/data',
          'Directory with the training data.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', '/tmp/log/train',
          'Directory with the log data.')
FLAGS = flags.FLAGS


def view_image(image_buffer):
  tf.image.decode_jpeg(image_buffer, channels=3)
  plt.imshow(image_buffer)
  plt.show()  

def train(train_dir, batch_size, num_batches, log_dir):
  images, labels = inputs(train_dir,
              batch_size,
              num_batches,
              one_hot_labels=True,
              train=True)

  predictions = simple(images)
  slim.losses.softmax_cross_entropy(predictions, labels)
  total_loss = tf.clip_by_value(slim.losses.get_total_loss(), 1e-10, 1.0)
  tf.scalar_summary('loss', total_loss)

  #optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
  optimizer = tf.train.GradientDescentOptimizer(0.005)
  train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

  slim.learning.train(train_op, log_dir, save_summaries_secs=20)


def main(unused_argv):
  train(FLAGS.train_dir, FLAGS.batch_size, FLAGS.num_batches, FLAGS.log_dir);

if __name__ == '__main__':
  tf.app.run()

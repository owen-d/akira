import tensorflow.contrib.slim as slim
import tensorflow as tf
from net import net

train_log_dir = '/tmp/log/train'
if not gfile.Exists(train_log_dir):
  gfile.MakeDirs(train_log_dir)

g = tf.Graph()

with g.as_default():
  # load
  images, labels = image_processing.distorted_inputs(
      dataset,
      num_preprocess_threads=num_preprocess_threads)

  # define model
  predictions = net(images, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)

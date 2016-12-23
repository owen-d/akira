import tensorflow.contrib.slim as slim
import tensorflow as tf
from net import simple
from dataset import FilmData
import image_processing


flags = tf.app.flags
# flags.DEFINE_string('train_dir', '/tmp/output',
  # 'Directory with the training data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', './log/train',
  'Directory with the log data.')
FLAGS = flags.FLAGS


tf.app.flags.DEFINE_string('train_dir', '/tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")


def train(train_dir, batch_size, num_batches, log_dir, dataset=FilmData('train')):
  # Calculate the learning rate schedule.
  num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                           FLAGS.batch_size)

  images, labels = image_processing.distorted_inputs(
      dataset)


  predictions = simple(images[0])

  slim.losses.softmax_cross_entropy(predictions, labels[0])
  total_loss = slim.losses.get_total_loss()
  tf.scalar_summary('loss', total_loss)

  optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
  train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

  for step in xrange(FLAGS.max_steps):
    start_time = time.time()
    _, loss_value = sess.run([train_op, total_loss])
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 10 == 0:
      examples_per_sec = FLAGS.batch_size / float(duration)
      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step, loss_value,
                          examples_per_sec, duration))

    if step % 100 == 0:
      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, step)

    # Save the model checkpoint periodically.
    if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)




def main(unused_argv):
  return train(FLAGS.train_dir, FLAGS.batch_size, FLAGS.num_batches, FLAGS.log_dir)

if __name__ == '__main__':
  tf.app.run()
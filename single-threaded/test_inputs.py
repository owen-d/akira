import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import preprocessing
from model import inputs

train_file = '/home/owen/projects/akira/build/single-threaded/train.tfrecords'


def view_image(image):
  print(image.shape)
  plt.imshow(image)
  plt.show()

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features={
    'image/encoded': tf.FixedLenFeature([], tf.string),
    'image/class/label': tf.FixedLenFeature([], tf.int64),
    'image/class/label': tf.FixedLenFeature([], tf.int64)
    })

  image = features['image/encoded']
  label = features['image/class/label']
  classification = features['image/class/label']
  return image, label, classification

def alt_loader(batch_size):
  images, labels = inputs('/home/owen/projects/akira/build/single-threaded',
              batch_size,
              None,
              one_hot_labels=True,
              train=True)
  return images, labels

def main():
  with tf.Session() as sess:
    #queue = filename_queue = tf.train.string_input_producer(
    #    [train_file], num_epochs=None)

    #image, label, classification  = read_and_decode(queue)
    images, labels = alt_loader(5)
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    images, labels = sess.run([images, labels])
    print(labels)
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  main()

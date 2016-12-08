import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import preprocessing

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

def main():
  with tf.Session() as sess:
    queue = filename_queue = tf.train.string_input_producer(
        [train_file], num_epochs=None)

    image, label, classification  = read_and_decode(queue)
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20):
      image_buffer, ex_label, ex_class  = sess.run([image, label, classification])
      #print(i, ex_label, ex_class)
      if i % 10 == 0:
        image_test = preprocessing.preprocess_image(image_buffer)
        result = sess.run(image_test)
        view_image(result)
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  main()

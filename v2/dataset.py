"""Small library that points to a data set.
Methods of Data class:
  data_files: Returns a python list of all (sharded) data set files.
  num_examples_per_epoch: Returns the number of examples in the data set.
  num_classes: Returns the number of classes in the data set.
  reader: Return a reader for a single entry from the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os


import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_string('data_dir', '/tmp/mydata',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")


class Dataset(object):
  """A simple class for handling data sets."""
  __metaclass__ = ABCMeta

  def __init__(self, name, subset):
    """Initialize dataset using a subset and the path to the data."""
    assert subset in self.available_subsets(), self.available_subsets()
    self.name = name
    self.subset = subset

  @abstractmethod
  def num_classes(self):
    """Returns the number of classes in the data set."""
    pass
    # return 10

  @abstractmethod
  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    pass
    # if self.subset == 'train':
    #   return 10000
    # if self.subset == 'validation':
    #   return 1000


  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'validation']

  def data_files(self):
    """Returns a python list of all (sharded) data subset files.
    Returns:
      python list of all (sharded) data set files.
    Raises:
      ValueError: if there are not data_files matching the subset.
    """
    tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % self.subset)
    print(tf_record_pattern)
    data_files = tf.gfile.Glob(tf_record_pattern)
    if not data_files:
      print('No files found for dataset %s/%s at %s' % (self.name,
                                                        self.subset,
                                                        FLAGS.data_dir))

      exit(-1)
    return data_files

  def reader(self):
    """Return a reader for a single entry from the data set.
    See io_ops.py for details of Reader class.
    Returns:
      Reader object that reads the data set.
    """
    return tf.TFRecordReader()


class FilmData(Dataset):
  """Flowers data set."""

  def __init__(self, subset):
    super(FilmData, self).__init__('Films', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 5

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 19244
    if self.subset == 'validation':
      return 8250

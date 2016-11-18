"""Small library that points to the film data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dataset import Dataset

# helpful script to find files
#for dir in `find build/TRAIN/* -maxdepth 0`; do echo "$(ls -1 $dir | wc -l)"; done | awk ' { sum += $1; } END { print sum; }'

class FilmData(Dataset):
  """Film data set."""

  def __init__(self, subset):
    super(FilmData, self).__init__('Film', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 5

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 19244
    if self.subset == 'validation':
      return 8250

  def download_message(self):
    """Instruction to download and extract the tarball from Film website."""

    print('Failed to find any Film %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('Please see README.md for instructions on how to build '
          'the Film dataset using download_and_preprocess_Film.\n')

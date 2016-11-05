# need a randomizer fn to randomly sort an array of strings (of file names) so that we can then feed sequentially into train/validation sets
# need a fn to create dirs and move selected files into validation/training sets (symlink)
# need a fn to select only a subset of images (i.e. module 24 for 1 image/sec)
import subprocess
import hashlib
import os
from file_utils import ensure_dir, clean_dir, get_files_in_dir
import glob
import math

def select_every(step, files):
  return zip(*filter(lambda x: x[0] % step == 0, enumerate(files)))[1]
# print select_every(2, [0, 1, 2, 3, 4, 5])

def get_num_images_in_dir(dir):
  cmd = 'ls -tr {} | wc -l'.format(dir)
  output = subprocess.check_output(cmd, shell=True)
  return int(output)

def randomize_order(items):
  def assign_hash_order(s):
    h = hashlib.sha1(s).hexdigest()[-5:]
    # return an in on the scale of (0, 16^5)
    return int(h, 16)

  return sorted(items, lambda x, y: assign_hash_order(x) - assign_hash_order(y))
# print randomize_order(['1', 'a', 'b', 'c', '3', 'fdas', 'fdsasd'])

def create_train_and_validation_dirs(path):
  train_path = os.path.join(path, 'TRAIN')
  validation_path = os.path.join(path, 'VALIDATION')

  for path in [train_path, validation_path]:
    ensure_dir(path);
    clean_dir(path);

def trim_edge_photos(dir, start=0.05, end=0.12):
  def str_to_int(s):
    return int(s)
  num_images = get_num_images_in_dir(dir)
  images = get_files_in_dir(dir)
  sorted_images = sorted(lambda x, y: str_to_int(x) - str_to_int(y))
  return sorted_images[math.floor(start * num_images):math.floor(num_images - (end * num_images))]


if __name__ == "__main__":
  classes_parent_dir = os.path.join(os.getenv('HOME'), 'block', 'film_images')
  classes_dirs = get_files_in_dir(classes_parent_dir)
  print classes_dirs
  # classes = ['Akira', 'BeautyAndTheBeast', 'Bladerunner', 'PorcoRosso', 'TronLegacy']
  # classes_dirs = map(lambda x: os.path.join(classes_parent_dir, x), classes)

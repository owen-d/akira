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

def numerical_order(items):
  def get_num(s):
    string = s.split('/')[-1].split('.')[0]
    return int(string)

  return sorted(items, lambda x, y: get_num(x) - get_num(y))

def create_train_and_validation_dirs(path):
  train_path = os.path.join(path, 'TRAIN')
  validation_path = os.path.join(path, 'VALIDATION')

  for path in [train_path, validation_path]:
    ensure_dir(path);
    clean_dir(path);
  print '''
  created train and validation dirs at:
  {}
  {}'''.format(train_path, validation_path)

def trim_edge_photos(dir, start=0.05, end=0.12):
  def str_to_int(s):
    return int(s)
  num_images = get_num_images_in_dir(dir)
  images = get_files_in_dir(dir)
  images = numerical_order(images)
  start_idx = int(math.floor(start * num_images))
  end_idx = int(math.floor(num_images - (end * num_images)))
  return images[start_idx:end_idx]

def link_images(image_list, target_dir):
  for image in image_list:
    stripped_name = image.split('/')[-1].split('.')[0]
    output_path = os.path.join(target_dir, stripped_name)
    os.symlink(image, output_path)

def build_train_and_validation_sets(source_dir, build_dir, train_ratio=0.7, frame_step=24):
  classification = source_dir.split('/')[-1]
  validation_dir = os.path.join(build_dir, 'VALIDATION', classification)
  train_dir = os.path.join(build_dir, 'TRAIN', classification)

  #ensure directory structure a la '${build_dir}/${VALIDATION || TRAIN}/${CLASSIFICATION}' exists
  ensure_dir(train_dir)
  ensure_dir(validation_dir)

  # now we need to pull images from source dir, order them numerically,
  # trim the edge percentages, select images at specified offsets, randomize them, and put them in respective dirs
  print 'processing image files for class: {}'.format(classification)
  image_files = trim_edge_photos(source_dir)
  image_files = select_every(frame_step, image_files)
  image_files = randomize_order(image_files)
  split_idx = int(math.floor(train_ratio * len(image_files)))
  link_images(image_files[:split_idx], train_dir)
  link_images(image_files[split_idx:], validation_dir)
  print 'finished {} train and {} validation images for class: {}'.format(split_idx, len(image_files) - split_idx, classification)


if __name__ == "__main__":
  try:
    classes_parent_dir = os.environ['CLASSES_PARENT_DIR']
  except:
    classes_parent_dir = os.path.join(os.getenv('HOME'), 'block', 'film_images')

  classes_dirs = get_files_in_dir(classes_parent_dir)
  classes = map(lambda x: x.split('/')[-1].split('.')[0], classes_dirs)

  try:
    build_dir = os.environ['BUILD_DIR']
  except:
    build_dir = os.path.join(os.getcwd(), 'build')

  #start fresh by removing build dir
  create_train_and_validation_dirs(build_dir)
  output_dir = os.path.join(build_dir, 'output')
  ensure_dir(output_dir)
  clean_dir(output_dir)

  for class_dir in classes_dirs:
    build_train_and_validation_sets(class_dir, build_dir)

  print '''
  train/validation sets available at:
    {}'''.format(build_dir)

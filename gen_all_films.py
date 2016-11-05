import gen_images
import glob
import os
from file_utils import ensure_dir

INPUT_DIR = os.path.join(os.getenv('HOME'), 'Downloads/films')
OUTPUT_DIR = os.path.join(os.getenv('HOME'), 'block/film_images')

ensure_dir(OUTPUT_DIR)

files = glob.glob(os.path.join(INPUT_DIR, '*'))

for file in files:
  print 'processing %s' % file
  film_out_dir = os.path.join(OUTPUT_DIR, file.split('.')[-2].split('/')[-1])
  # print file, film_out_dir, file.split('.')[-2].split('/')[-1]
  gen_images.gen_images_from_film(file, film_out_dir)
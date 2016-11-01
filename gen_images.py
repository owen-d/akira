import cv2
import os
import glob

def ensure_dir(d):
  if not os.path.exists(d):
    os.makedirs(d)

def clean_dir(d):
  if os.path.exists(d):
    files = glob.glob(os.path.join(d, '*'))
    for f in files:
      os.remove(f)

def gen_images_from_film(path, out_dir='./data/frames'):
  ensure_dir(out_dir)
  vidcap = cv2.VideoCapture(path)

  success, image = vidcap.read()

  count = 0

  while success:
    success, image = vidcap.read()
    cv2.imwrite(out_dir + '/frame-%d.jpg' % count, image)
    count +=1

  print 'finished creating %d frames' % count

def select_by_step(prefix_dir='data/frames', prefix_name='akira-frame-', start=2000, end=175000, step=24, suffix='.jpg'):
  filenames = []
  cwd = os.getcwd()
  for x in xrange(start, end, step):
    name = '{}{}{}'.format(prefix_name, x, suffix)
    name = os.path.join(cwd, prefix_dir, name)
    if os.path.exists(name):
      filenames.append(name)

  return filenames

def symlink_images(image_filenames, symlink_dir='/tmp/categories', category='akira', suffix=None, report_every=100):
  category_dir = os.path.join(symlink_dir, category)
  clean_dir(category_dir)
  ensure_dir(category_dir)

  for idx, source in enumerate(image_filenames):
    suffix = suffix or '.' + source.split('.')[-1]
    output_path = os.path.join(symlink_dir, category, '{}{}'.format(idx, suffix))
    os.symlink(source, output_path) 


# if __name__ == "__main__":
#   steps = select_by_step(step=600)
#   symlink_images(steps, symlink_dir=os.environ['BUILDING_DIR'] + '/categories')
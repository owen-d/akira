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

def get_files_in_dir(d):
  search_path = os.path.join(d, '*')
  if os.path.exists(d):
    files = glob.glob(search_path)
    return files

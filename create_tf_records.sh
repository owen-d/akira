#!/bin/bash
set -e
#script should be run from a https://github.com/tensorflow/models/tree/master/inception installation.


# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=$HOME/projects/akira/build/output
TRAIN_DIR="$HOME/projects/akira/build/TRAIN"
VALIDATION_DIR="$HOME/projects/akira/build/VALIDATION"
LABELS_FILE="$HOME/projects/akira/build/labels.txt"

#build labels file
echo $TRAIN_DIR
ls $TRAIN_DIR -1 > $LABELS_FILE

# build the preprocessing script.
bazel build inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data \
  --train_directory="${TRAIN_DIR}" \
  --validation_directory="${VALIDATION_DIR}" \
  --output_directory="${OUTPUT_DIRECTORY}" \
  --labels_file="${LABELS_FILE}" \
  --train_shards=128 \
  --validation_shards=24 \
  --num_threads=8
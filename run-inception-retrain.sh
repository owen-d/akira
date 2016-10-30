set -e


TF_DIR=~/projects/tensorflow
export BUILDING_DIR=./data/retrain-inception

# populate our categories
python gen_images.py

# retrain
$TF_DIR/bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir=$BUILDING_DIR/categories \
  --bottleneck_dir=$BUILDING_DIR/bottleneck
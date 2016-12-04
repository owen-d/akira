from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_integer('image_size', 299,
  """image size for square construction""")

FLAGS = tf.app.flags.FLAGS



def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.
  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

def preprocess_image(image_buffer, train=True):
  """Decode and preprocess one image for evaluation or training.
  Args:
    image_buffer: JPEG encoded string Tensor
    train: boolean
  Returns:
    3-D float Tensor containing an appropriately scaled image
  Raises:
    ValueError: if user does not provide bounding box
  """

  image = decode_jpeg(image_buffer)
  height = FLAGS.image_size
  width = FLAGS.image_size

  # Resize the image to the original height and width.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [height, width],
                                   align_corners=False)
  image = tf.squeeze(image, [0])

  # Finally, rescale to [-1,1] instead of [0, 1)
  image = tf.sub(image, 0.5)
  image = tf.mul(image, 2.0)
  return image

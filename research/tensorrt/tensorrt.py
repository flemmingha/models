# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Methods for running the Official Models with TensorRT.

Please note that all of these methods are in development, and subject to
rapid change.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imghdr
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader

from official.resnet import imagenet_preprocessing

_GPU_MEM_FRACTION = 0.50


################################################################################
# Utils for converting a SavedModel to a Frozen Graph.
################################################################################
def get_serving_meta_graph_def(savedmodel_dir):
  """Extract the SERVING MetaGraphDef from a SavedModel directory.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.

  Returns:
    MetaGraphDef that should be used for tag_constants.SERVING mode.

  Raises:
    ValueError: if a MetaGraphDef matching tag_constants.SERVING is not found.
  """
  # We only care about the serving graph def
  tag_set = set([tf.saved_model.tag_constants.SERVING])
  serving_graph_def = None
  saved_model = reader.read_saved_model(savedmodel_dir)
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == tag_set:
      serving_graph_def = meta_graph_def
  if not serving_graph_def:
    raise ValueError("No MetaGraphDef found for tag_constants.SERVING. "
                     "Please make sure the SavedModel includes a SERVING def.")

  return serving_graph_def


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_dir):
  """Convert a SavedModel to a Frozen Graph.

  A SavedModel includes a `variables` directory with variable values,
  and a specification of the MetaGraph in a ProtoBuffer file. A Frozen Graph
  takes the variable values and inserts them into the graph, such that the
  SavedModel is all bundled into a single file. TensorRT and TFLite both
  leverage Frozen Graphs. Here, we provide a simple utility for converting
  a SavedModel into a frozen graph for use with these other tools.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.
    output_dir: string representing path to the output directory for saving
      the frozen graph.

  Returns:
    Frozen Graph definition for use.
  """
  meta_graph = get_serving_meta_graph_def(savedmodel_dir)
  signature_def = tf.contrib.saved_model.get_signature_def_by_key(
      meta_graph,
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

  outputs = [v.name for v in signature_def.outputs.itervalues()]
  output_names = [node.split(":")[0] for node in outputs]

  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:

    tf.saved_model.loader.load(
        sess, meta_graph.meta_info_def.tags, savedmodel_dir)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), output_names)

  output_path = os.path.join(output_dir, "frozen_graph.pb")
  with tf.gfile.GFile(output_path, "wb") as f:
    f.write(frozen_graph_def.SerializeToString())

  return frozen_graph_def


def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


################################################################################
# Prep the image input to TensorRT.
################################################################################
def preprocess_image(file_name, output_height=224, output_width=224,
                     num_channels=3):
  """Run standard ImageNet preprocessing on the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    output_height: int, final height of image
    output_width: int, final width of image
    num_channels: int, depth of input image

  Returns:
    Float array representing processed image with shape
      [output_height, output_width, num_channels]

  Raises:
    ValueError: if image is not a JPEG.
  """
  if imghdr.what(file_name) != "jpeg":
    raise ValueError("At this time, only JPEG images are supported. "
                     "Please try another image.")

  image_buffer = tf.read_file(file_name)
  normalized = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=None,
      output_height=output_height,
      output_width=output_width,
      num_channels=num_channels,
      is_training=False)

  with tf.Session() as sess:
    result = sess.run([normalized])

  return result[0]


def batch_from_image(file_name, batch_size, output_height=224, output_width=224,
                     num_channels=3):
  """Produce a batch of data from the passed image file.

  Args:
    file_name: string, path to file containing a JPEG image
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of input data

  Returns:
    Float array representing copies of the image with shape
      [batch_size, output_height, output_width, num_channels]
  """
  image_array = preprocess_image(
      file_name, output_height, output_width, num_channels)

  tiled_array = np.tile(image_array, [batch_size, 1, 1, 1])
  return tiled_array


def batch_from_random(batch_size, output_height=224, output_width=224,
                      num_channels=3):
  """Produce a batch of random data.

  Args:
    batch_size: int, the size of the desired batch of data
    output_height: int, final height of data
    output_width: int, final width of data
    num_channels: int, depth of output data

  Returns:
    Float array containing random numbers with shape
      [batch_size, output_height, output_width, num_channels]
  """
  shape = [batch_size, output_height, output_width, num_channels]
  return np.random.random_sample(shape)





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
"""Convenience functions for exporting models as SavedModels or other types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32,
                                           batch_size=1):
  """Returns a input_receiver_fn that can be used during serving.

  This expects examples to come through as float tensors, and simply
  wraps them as TensorServingInputReceivers.

  Arguably, this should live in tf.estimator.export. Testing here first.

  Args:
    shape: list representing target size of a single example.
    dtype: the expected datatype for the input example
    batch_size: number of input tensors that will be passed for prediction

  Returns:
    A function that itself returns a TensorServingInputReceiver.
  """
  def serving_input_receiver_fn():
    # Prep a placeholder where the input example will be fed in
    features = tf.placeholder(
        dtype=dtype, shape=[batch_size] + shape, name='input_tensor')

    return tf.estimator.export.TensorServingInputReceiver(
        features=features, receiver_tensors=features)

  return serving_input_receiver_fn


def convert_savedmodel_to_frozen_graph(savedmodel_dir, output_path):
  """Convert a SavedModel to a Frozen Graph.

  A SavedModel includes a `variables` directory with variable values,
  and a specification of the graph in a ProtoBuffer file. A Frozen Graph takes
  the variable values and inserts them into the graph, such that the
  SavedModel is all bundled into a single file. TensorRT and TFLite both
  leverage Frozen Graphs. Here, we provide a simple utility for converting
  a SavedModel into a frozen graph for use with these other tools.

  Args:
    savedmodel_dir: the string path to the directory containing the .pb
      and variables for a SavedModel. This is equivalent to the subdirectory
      that is created under the directory specified by --export_dir when
      running an Official Model.
    output_path: string representing the full path to the saved frozen graph.
      For example, `'my/path/frozen_graph.pb'`

  """



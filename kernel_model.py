# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kernel model.

This model allows interactions between the two inputs, such as attention.
"""
import sys

import tensorflow as tf
import tensorflow.contrib.learn as learn

import tf_utils
from common_model import embedding_layer


def kernel_model(features, mode, params, scope=None):
  """Kernel models that allow interaction between question and context.

  This is handler for all kernel models in this script. Models are called via
  `params.model_id` (e.g. `params.model_id = m00`).

  Function requirement for each model is in:
  https://www.tensorflow.org/extend/estimators

  This function does not have any dependency on FLAGS. All parameters must be
  passed through `params` argument.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable name scope.
  Returns:
    `(logits_start, logits_end, tensors)` pair. Tensors is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """
  this_module = sys.modules[__name__]
  model_fn = getattr(this_module, '_model_%s' % params.model_id)
  return model_fn(
      features, mode, params, scope=scope)


def _model_m00(features, mode, params, scope=None):
  """Simplified BiDAF, reaching 74~75% F1.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable name scope.
  Returns:
    `(logits_start, logits_end, tensors)` pair. Tensors is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """
  with tf.variable_scope(scope or 'kernel_model'):
    training = mode == learn.ModeKeys.TRAIN
    tensors = {}

    x, q = embedding_layer(features, mode, params)

    x0 = tf_utils.bi_rnn(
        params.hidden_size,
        x,
        sequence_length_list=features['context_num_words'],
        scope='x_bi_rnn_0',
        training=training,
        dropout_rate=params.dropout_rate)

    q0 = tf_utils.bi_rnn(
        params.hidden_size,
        q,
        sequence_length_list=features['question_num_words'],
        scope='q_bi_rnn_0',
        training=training,
        dropout_rate=params.dropout_rate)

    xq = tf_utils.att2d(
        q0,
        x0,
        mask=features['question_num_words'],
        tensors=tensors,
        scope='xq')
    xq = tf.concat([x0, xq, x0 * xq], 2)
    x1 = tf_utils.bi_rnn(
        params.hidden_size,
        xq,
        sequence_length_list=features['context_num_words'],
        training=training,
        scope='x1_bi_rnn',
        dropout_rate=params.dropout_rate)
    x2 = tf_utils.bi_rnn(
        params.hidden_size,
        x1,
        sequence_length_list=features['context_num_words'],
        training=training,
        scope='x2_bi_rnn',
        dropout_rate=params.dropout_rate)
    x3 = tf_utils.bi_rnn(
        params.hidden_size,
        x2,
        sequence_length_list=features['context_num_words'],
        training=training,
        scope='x3_bi_rnn',
        dropout_rate=params.dropout_rate)

    logits_start = tf_utils.exp_mask(
        tf.squeeze(
            tf.layers.dense(tf.concat([x1, x2], 2), 1, name='logits1'), 2),
        features['context_num_words'])
    logits_end = tf_utils.exp_mask(
        tf.squeeze(
            tf.layers.dense(tf.concat([x1, x3], 2), 1, name='logits2'), 2),
        features['context_num_words'])

    return logits_start, logits_end, tensors

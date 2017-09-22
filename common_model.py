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
"""Common components for feature and kernel models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn
import squad_data
import tf_utils

# This value is needed to compute best answer span with GPUs.
# Set this to a high value for large context, but note that it will take
# GPU memory.
MAX_CONTEXT_SIZE = 2000


def embedding_layer(features, mode, params, reuse=False):
  """Common embedding layer for feature and kernel functions.

  Args:
    features: A dictionary containing features, directly copied from `model_fn`.
    mode: Mode.
    params: Contains parameters, directly copied from `model_fn`.
    reuse: Reuse variables.
  Returns:
    `(x, q)` where `x` is embedded representation of context, and `q` is the
    embedded representation of the question.
  """
  with tf.variable_scope('embedding_layer', reuse=reuse):
    training = mode == learn.ModeKeys.TRAIN
    with tf.variable_scope('embedding'):
      char_emb_mat = tf.get_variable('char_emb_mat',
                                     [params.char_vocab_size, params.emb_size])
      xc = tf.nn.embedding_lookup(char_emb_mat,
                                  features['indexed_context_chars'])
      qc = tf.nn.embedding_lookup(char_emb_mat,
                                  features['indexed_question_chars'])

      xc = tf.reduce_max(xc, 2)
      qc = tf.reduce_max(qc, 2)

      _, xv, qv = glove_layer(features)

      # Concat
      x = tf.concat([xc, xv], 2)
      q = tf.concat([qc, qv], 2)

    x = tf_utils.highway_net(
        x, 2, training=training, dropout_rate=params.dropout_rate)
    q = tf_utils.highway_net(
        q, 2, training=training, dropout_rate=params.dropout_rate, reuse=True)

  return x, q


def glove_layer(features, scope=None):
  """GloVe embedding layer.

  The first two words of `features['emb_mat']` are <PAD> and <UNK>.
  The other words are actual words. So we learn the representations of the
  first two words but the representation of other words are fixed (GloVe).

  Args:
    features: `dict` of feature tensors.
    scope: `str` for scope name.
  Returns:
    A tuple of tensors, `(glove_emb_mat, context_emb, question_emb)`.
  """
  with tf.variable_scope(scope or 'glove_layer'):
    glove_emb_mat_const = tf.slice(features['emb_mat'], [2, 0], [-1, -1])
    glove_emb_mat_var = tf.get_variable('glove_emb_mat_var',
                                        [2,
                                         glove_emb_mat_const.get_shape()[1]])
    glove_emb_mat = tf.concat([glove_emb_mat_var, glove_emb_mat_const], 0)
    xv = tf.nn.embedding_lookup(glove_emb_mat,
                                features['glove_indexed_context_words'])
    qv = tf.nn.embedding_lookup(glove_emb_mat,
                                features['glove_indexed_question_words'])
    return glove_emb_mat, xv, qv


def char_layer(features, params, scope=None):
  """Character embedding layer.

  Args:
    features: `dict` of feature tensors.
    params: `HParams` object.
    scope: `str` for scope name.
  Returns:
    a tuple of tensors, `(char_emb_mat, context_emb, question_emb)`.
  """
  with tf.variable_scope(scope or 'char_layer'):
    char_emb_mat = tf.get_variable('char_emb_mat',
                                   [params.char_vocab_size, params.emb_size])
    xc = tf.nn.embedding_lookup(char_emb_mat, features['indexed_context_chars'])
    qc = tf.nn.embedding_lookup(char_emb_mat,
                                features['indexed_question_chars'])

    xc = tf.reduce_max(xc, 2)
    qc = tf.reduce_max(qc, 2)
    return char_emb_mat, xc, qc


def get_pred_ops(features, params, logits_start, logits_end, no_answer_bias):
  """Get prediction op dictionary given start & end logits.

  This dictionary will contain predictions as well as everything needed
  to produce the nominal answer and identifier (ids).

  Args:
    features: Features.
    params: `HParams` object.
    logits_start: [batch_size, context_size]-shaped tensor of logits for start.
    logits_end: Similar to `logits_start`, but for end. This tensor can be also
      [batch_size, context_size, context_size], in which case the true answer
      start is used to index on dim 1 (context_size).
    no_answer_bias: [batch_size, 1]-shaped tensor, bias for no answer decision.
  Returns:
    A dictionary of prediction tensors.
  """
  max_x_len = tf.shape(logits_start)[1]

  if len(logits_end.get_shape()) == 3:
    prob_end_given_start = tf.nn.softmax(logits_end)
    prob_start = tf.nn.softmax(logits_start)
    prob_start_end = prob_end_given_start * tf.expand_dims(prob_start, -1)

    upper_tri_mat = tf.slice(
        np.triu(
            np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32') -
            np.triu(
                np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32'),
                k=params.max_answer_size)), [0, 0], [max_x_len, max_x_len])
    prob_start_end *= tf.expand_dims(upper_tri_mat, 0)

    prob_end = tf.reduce_sum(prob_start_end, 1)
    answer_pred_start = tf.argmax(tf.reduce_max(prob_start_end, 2), 1)
    answer_pred_end = tf.argmax(tf.reduce_max(prob_start_end, 1), 1)
    answer = squad_data.get_answer_op(features['context'],
                                      features['context_words'],
                                      answer_pred_start, answer_pred_end)
    answer_prob = tf.reduce_max(prob_start_end, [1, 2])

    predictions = {
        'yp1': answer_pred_start,
        'yp2': answer_pred_end,
        'p1': prob_start,
        'p2': prob_end,
        'a': answer,
        'id': features['id'],
        'context': features['context'],
        'context_words': features['context_words'],
        'answer_prob': answer_prob,
        'has_answer': answer_prob > 0.0,
    }

  else:
    # Predictions and metrics.
    concat_logits_start = tf.concat([no_answer_bias, logits_start], 1)
    concat_logits_end = tf.concat([no_answer_bias, logits_end], 1)

    concat_prob_start = tf.nn.softmax(concat_logits_start)
    concat_prob_end = tf.nn.softmax(concat_logits_end)

    no_answer_prob_start = tf.squeeze(
        tf.slice(concat_prob_start, [0, 0], [-1, 1]), 1)
    no_answer_prob_end = tf.squeeze(
        tf.slice(concat_prob_end, [0, 0], [-1, 1]), 1)
    no_answer_prob = no_answer_prob_start * no_answer_prob_end
    has_answer = no_answer_prob < 0.5
    prob_start = tf.slice(concat_prob_start, [0, 1], [-1, -1])
    prob_end = tf.slice(concat_prob_end, [0, 1], [-1, -1])

    # This is only for computing span accuracy and not used for training.
    # Masking with `upper_triangular_matrix` only allows valid spans,
    # i.e. `answer_pred_start` <= `answer_pred_end`.
    # TODO(seominjoon): Replace with dynamic upper triangular matrix.
    upper_tri_mat = tf.slice(
        np.triu(
            np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32') -
            np.triu(
                np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32'),
                k=params.max_answer_size)), [0, 0], [max_x_len, max_x_len])
    prob_mat = tf.expand_dims(prob_start, -1) * tf.expand_dims(
        prob_end, 1) * tf.expand_dims(upper_tri_mat, 0)
    # TODO(seominjoon): Handle this.
    logits_mat = tf_utils.exp_mask(
        tf.expand_dims(logits_start, -1) + tf.expand_dims(logits_end, 1),
        tf.expand_dims(upper_tri_mat, 0),
        mask_is_length=False)
    del logits_mat

    answer_pred_start = tf.argmax(tf.reduce_max(prob_mat, 2), 1)
    answer_pred_end = tf.argmax(tf.reduce_max(prob_mat, 1), 1)  # [batch_size]
    answer = squad_data.get_answer_op(features['context'],
                                      features['context_words'],
                                      answer_pred_start, answer_pred_end)
    answer_prob = tf.reduce_max(prob_mat, [1, 2])

    predictions = {
        'yp1': answer_pred_start,
        'yp2': answer_pred_end,
        'p1': prob_start,
        'p2': prob_end,
        'a': answer,
        'id': features['id'],
        'context': features['context'],
        'context_words': features['context_words'],
        'no_answer_prob': no_answer_prob,
        'no_answer_prob_start': no_answer_prob_start,
        'no_answer_prob_end': no_answer_prob_end,
        'answer_prob': answer_prob,
        'has_answer': has_answer,
    }
  return predictions


def get_loss(answer_start,
             answer_end,
             logits_start,
             logits_end,
             no_answer_bias,
             sparse=True):
  """Get loss given answer and logits.

  Args:
    answer_start: [batch_size, num_answers] shaped tensor if `sparse=True`, or
    [batch_size, context_size] shaped if `sparse=False`.
    answer_end: Similar to `answer_start` but for end.
    logits_start: [batch_size, context_size]-shaped tensor for answer start
      logits.
    logits_end: Similar to `logits_start`, but for end. This tensor can be also
      [batch_size, context_size, context_size], in which case the true answer
      start is used to index on dim 1 (context_size).
    no_answer_bias: [batch_size, 1] shaped tensor, bias for no answer decision.
    sparse: Indicates whether `answer_start` and `answer_end` are sparse or
      dense.
  Returns:
    Float loss tensor.
  """
  if sparse:
    # During training, only one answer. During eval, multiple answers.
    # TODO(seominjoon): Make eval loss minimum over multiple answers.
    # Loss for start.
    answer_start = tf.squeeze(tf.slice(answer_start, [0, 0], [-1, 1]), 1)
    answer_start += 1
    logits_start = tf.concat([no_answer_bias, logits_start], 1)
    losses_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer_start, logits=logits_start)
    loss_start = tf.reduce_mean(losses_start)
    tf.add_to_collection('losses', loss_start)

    # Loss for end.
    answer_end = tf.squeeze(tf.slice(answer_end, [0, 0], [-1, 1]), 1)
    # Below are start-conditional loss, where every start position has its
    # own logits for end position.
    if len(logits_end.get_shape()) == 3:
      mask = tf.one_hot(answer_start, tf.shape(logits_end)[1])
      mask = tf.cast(tf.expand_dims(mask, -1), 'float')
      logits_end = tf.reduce_sum(mask * logits_end, 1)
    answer_end += 1
    logits_end = tf.concat([no_answer_bias, logits_end], 1)
    losses_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=answer_end, logits=logits_end)
    loss_end = tf.reduce_mean(losses_end)
    tf.add_to_collection('losses', loss_end)
  else:
    # TODO(seominjoon): Implement no answer capability for non sparse labels.
    losses_start = tf.nn.softmax_cross_entropy_with_logits(
        labels=answer_start, logits=logits_start)
    loss_start = tf.reduce_mean(losses_start)
    tf.add_to_collection('losses', loss_start)

    losses_end = tf.nn.softmax_cross_entropy_with_logits(
        labels=answer_end, logits=logits_end)
    loss_end = tf.reduce_mean(losses_end)
    tf.add_to_collection('losses', loss_end)

  return tf.add_n(tf.get_collection('losses'))


def get_train_op(loss,
                 var_list=None,
                 post_ops=None,
                 inc_step=True,
                 learning_rate=0.001,
                 clip_norm=0.0):
  """Get train op for the given loss.

  Args:
    loss: Loss tensor.
    var_list: A list of variables that the train op will minimize.
    post_ops: A list of ops that will be run after the train op. If not defined,
      no op is run after train op.
    inc_step: If `True`, will increase the `global_step` variable by 1 after
      step.
    learning_rate: Initial learning rate for the optimizer.
    clip_norm: If specified, clips the gradient of each variable by this value.
  Returns:
    Train op to be used for training.
  """

  global_step = tf.train.get_global_step() if inc_step else None
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  grads = optimizer.compute_gradients(loss, var_list=var_list)
  grads = [(grad, var) for grad, var in grads if grad is not None]
  for grad, var in grads:
    tf.summary.histogram(var.op.name, var)
    tf.summary.histogram('gradients/' + var.op.name, grad)
  if clip_norm:
    grads = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grads]
  train_op = optimizer.apply_gradients(grads, global_step=global_step)

  if post_ops is not None:
    with tf.control_dependencies([train_op]):
      train_op = tf.group(*post_ops)

  return train_op

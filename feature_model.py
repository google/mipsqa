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
"""Feature map model.

Obtains separate embedding for question and each word of context, and then
use distance metric (dot, l1, l2) to obtain the closest word in the context.
"""
# TODO(seominjoon): Refactor file and function names (e.g. drop squad).
import sys

import tensorflow as tf
import tensorflow.contrib.learn as learn

import tf_utils
from common_model import char_layer
from common_model import embedding_layer
from common_model import glove_layer


def feature_model(features, mode, params, scope=None):
  """Handler for SQuAD feature models: only allow feature mapping.

  Every feature model is called via this function. Which model to call can
  be controlled via `params.model_id` (e.g. `params.model_id = m00`).

  Function requirement is in:
  https://www.tensorflow.org/extend/estimators

  This function does not have any dependency on FLAGS. All parameters must be
  passed through `params` argument for all models in this script.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """
  this_module = sys.modules[__name__]
  model_fn = getattr(this_module, '_model_%s' % params.model_id)
  return model_fn(features, mode, params, scope=scope)


def _get_logits_from_multihead_x_and_q(x_start,
                                       x_end,
                                       q_start,
                                       q_end,
                                       x_len,
                                       dist,
                                       x_start_reduce_fn=None,
                                       x_end_reduce_fn=None,
                                       q_start_reduce_fn=None,
                                       q_end_reduce_fn=None):
  """Helper function for getting logits from context and question vectors.

  Args:
    x_start: [batch_size, context_num_words, hidden_size]-shaped float tensor,
      representing context vectors for answer start.
      Can have one more dim at the end, which will be reduced after computing
      distance via `reduce_fn`.
    x_end: [batch_size, context_num_words, hidden_size]-shaped float tensor,
      representing context vectors for answer end.
      Can have one more dim at the end, which will be reduced after computing
      distance via `reduce_fn`.
    q_start: [batch_size, hidden_size]-shaped float tensor,
      representing question vector for answer start.
      Can have one more dim at the end, which will be reduced after computing
      distance via `reduce_fn`.
    q_end: [batch_size, hidden_size]-shaped float tensor,
      representing question vector for answer end.
      Can have one more dim at the end, which will be reduced after computing
      distance via `reduce_fn`.
    x_len: [batch_size]-shaped int64 tensor, containing length of each context.
    dist: distance function, `dot`, `l1`, or `l2`.
    x_start_reduce_fn: reduction function that takes in the tensor as first
      argument and the axis as the second argument. Default is `tf.reduce_max`.
      Reduction for `x_start` and `x_end` if the extra dim is provided.
      Note that `l1` and `l2` distances are first negated and then the reduction
      is applied, so that `reduce_max` effectively gets minimal distance.
      Note that, for correct performance during inference when using nearest
      neighbor, reduction must be the default one (None, i.e. `tf.reduce_max`).
    x_end_reduce_fn: ditto, for end.
    q_start_reduce_fn: reduction function that takes in the tensor as first
      argument and the axis as the second argument. Default is `tf.reduce_max`.
      Reduction for `q_start` and `q_end` if the extra dim is provided.
      Note that `l1` and `l2` distances are first negated and then the reduction
      is applied, so that `reduce_max` effectively gets minimal distance.
      This can be any reduction function, unlike `x_reduce_fn`.
    q_end_reduce_fn: ditto, for end.

  Returns:
    a tuple `(logits_start, logits_end)` where each tensor's shape is
    [batch_size, context_num_words].

  """
  if len(q_start.get_shape()) == 1:
    # q can be universal, e.g. trainable weights.
    q_start = tf.expand_dims(q_start, 0)
    q_end = tf.expand_dims(q_end, 0)

  # Expand q first to broadcast for `context_num_words` dim.
  q_start = tf.expand_dims(q_start, 1)
  q_end = tf.expand_dims(q_end, 1)

  # Add one dim at the end if no additional dim at the end, to make them rank-4.
  if len(x_start.get_shape()) == 3:
    x_start = tf.expand_dims(x_start, -1)
    x_end = tf.expand_dims(x_end, -1)
  if len(q_start.get_shape()) == 3:
    q_start = tf.expand_dims(q_start, -1)
    q_end = tf.expand_dims(q_end, -1)

  # Add dim to outer-product x and q. This makes them rank-5 tensors.
  # shape : [batch_size, context_words, hidden_size, num_heads, 1]
  x_start = tf.expand_dims(x_start, -1)
  x_end = tf.expand_dims(x_end, -1)

  # shape : [batch_size, context_words, hidden_size, 1, num_heads]
  q_start = tf.expand_dims(q_start, 3)
  q_end = tf.expand_dims(q_end, 3)

  if x_start_reduce_fn is None:
    x_start_reduce_fn = tf.reduce_max
  if x_end_reduce_fn is None:
    x_end_reduce_fn = tf.reduce_max
  if q_start_reduce_fn is None:
    q_start_reduce_fn = tf.reduce_max
  if q_end_reduce_fn is None:
    q_end_reduce_fn = tf.reduce_max

  if dist == 'dot':
    logits_start = q_start_reduce_fn(
        x_start_reduce_fn(tf.reduce_sum(x_start * q_start, 2), 2), 2)
    logits_end = q_end_reduce_fn(
        x_end_reduce_fn(tf.reduce_sum(x_end * q_end, 2), 2), 2)
  elif dist == 'l1':
    logits_start = q_start_reduce_fn(
        x_start_reduce_fn(
            -tf.norm(x_start - q_start, ord=1, axis=2, keep_dims=True), 2), 2)
    logits_start = tf.squeeze(
        tf.layers.dense(logits_start, 1, name='logits_start'), 2)
    logits_end = q_end_reduce_fn(
        x_end_reduce_fn(-tf.norm(x_end - q_end, ord=1, axis=2, keep_dims=True),
                        2), 2)
    logits_end = tf.squeeze(
        tf.layers.dense(logits_end, 1, name='logits_end'), 2)
  elif dist == 'l2':
    logits_start = q_start_reduce_fn(
        x_start_reduce_fn(
            -tf.norm(x_start - q_start, ord=2, axis=2, keep_dims=True), 2), 2)
    logits_start = tf.squeeze(
        tf.layers.dense(logits_start, 1, name='logits_start'), 2)
    logits_end = q_end_reduce_fn(
        x_end_reduce_fn(-tf.norm(x_end - q_end, ord=2, axis=2, keep_dims=True),
                        2), 2)
    logits_end = tf.squeeze(
        tf.layers.dense(logits_end, 1, name='logits_end'), 2)

  logits_start = tf_utils.exp_mask(logits_start, x_len)
  logits_end = tf_utils.exp_mask(logits_end, x_len)

  return logits_start, logits_end


def _model_m00(features, mode, params, scope=None):
  """LSTM-based model.

  This model uses two stacked LSTMs to output vectors for context, and
  self-attention to output vectors for question. This model reaches 57~58% F1.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope, default is `feature_model`.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """

  with tf.variable_scope(scope or 'feature_model'):
    training = mode == learn.ModeKeys.TRAIN

    x, q = embedding_layer(features, mode, params)

    x1 = tf_utils.bi_rnn(
        params.hidden_size,
        x,
        sequence_length_list=features['context_num_words'],
        scope='x_bi_rnn_1',
        training=training,
        dropout_rate=params.dropout_rate)

    x2 = tf_utils.bi_rnn(
        params.hidden_size,
        x1,
        sequence_length_list=features['context_num_words'],
        scope='x_bi_rnn_2',
        training=training,
        dropout_rate=params.dropout_rate)

    q1 = tf_utils.bi_rnn(
        params.hidden_size,
        q,
        sequence_length_list=features['question_num_words'],
        scope='q_bi_rnn_1',
        training=training,
        dropout_rate=params.dropout_rate)

    q2 = tf_utils.bi_rnn(
        params.hidden_size,
        q1,
        sequence_length_list=features['question_num_words'],
        scope='q_bi_rnn_2',
        training=training,
        dropout_rate=params.dropout_rate)

    # Self-attention to obtain single vector representation.
    q_start = tf_utils.self_att(
        q1, mask=features['question_num_words'], scope='q_start')
    q_end = tf_utils.self_att(
        q2, mask=features['question_num_words'], scope='q_end')

    logits_start, logits_end = _get_logits_from_multihead_x_and_q(
        x1, x2, q_start, q_end, features['context_num_words'], params.dist)
    return logits_start, logits_end, dict()


def _model_m01(features, mode, params, scope=None):
  """Self-attention with MLP, reaching 55~56% F1.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope, default is `feature_model`.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """

  with tf.variable_scope(scope or 'feature_model'):
    training = mode == learn.ModeKeys.TRAIN
    tensors = {}

    x, q = embedding_layer(features, mode, params)

    x0 = tf_utils.bi_rnn(
        params.hidden_size,
        x,
        sequence_length_list=features['context_num_words'],
        scope='bi_rnn_x0',
        training=training,
        dropout_rate=params.dropout_rate)

    x1 = tf_utils.bi_rnn(
        params.hidden_size,
        x0,
        sequence_length_list=features['context_num_words'],
        scope='bi_rnn_x1',
        training=training,
        dropout_rate=params.dropout_rate)

    x1 += x0

    q1 = tf_utils.bi_rnn(
        params.hidden_size,
        q,
        sequence_length_list=features['question_num_words'],
        scope='bi_rnn_q1',
        training=training,
        dropout_rate=params.dropout_rate)

    def get_x(x_, scope=None):
      with tf.variable_scope(scope or 'get_x_clue'):
        hidden_sizes = [params.hidden_size, params.hidden_size]
        attender = tf_utils.mlp(
            x_,
            hidden_sizes,
            activate_last=False,
            training=training,
            dropout_rate=params.dropout_rate,
            scope='attender')
        attendee = tf_utils.mlp(
            x_,
            hidden_sizes,
            activate_last=False,
            training=training,
            dropout_rate=params.dropout_rate,
            scope='attendee')
        clue = tf_utils.att2d(
            attendee,
            attender,
            a_val=x_,
            mask=features['context_num_words'],
            logit_fn='dot',
            tensors=tensors)
        return tf.concat([x_, clue], 2)

    x_start = get_x(x1, scope='get_x_start')
    x_end = get_x(x1, scope='get_x_end')

    q_type = tf_utils.self_att(
        q1,
        mask=features['question_num_words'],
        scope='self_att_q_type',
        tensors=tensors)
    q_clue = tf_utils.self_att(
        q1,
        mask=features['question_num_words'],
        scope='self_att_q_clue',
        tensors=tensors)
    q_start = q_end = tf.concat([q_type, q_clue], 1)

    logits_start, logits_end = _get_logits_from_multihead_x_and_q(
        x_start, x_end, q_start, q_end, features['context_num_words'],
        params.dist)
    return logits_start, logits_end, tensors


def _model_m02(features, mode, params, scope=None):
  """Self-attention with LSTM, reaching 59~60% F1.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope, default is `feature_model`.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """

  with tf.variable_scope(scope or 'feature_model'):
    training = mode == learn.ModeKeys.TRAIN
    tensors = {}

    x, q = embedding_layer(features, mode, params)

    x1 = tf_utils.bi_rnn(
        params.hidden_size,
        x,
        sequence_length_list=features['context_num_words'],
        scope='bi_rnn_x1',
        training=training,
        dropout_rate=params.dropout_rate)

    def get_clue(x_, scope=None):
      with tf.variable_scope(scope or 'get_clue'):
        attendee = tf_utils.bi_rnn(
            params.hidden_size,
            x_,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attendee',
            training=training,
            dropout_rate=params.dropout_rate)
        attender = tf_utils.bi_rnn(
            params.hidden_size,
            x_,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attender',
            training=training,
            dropout_rate=params.dropout_rate)
        clue = tf_utils.att2d(
            attendee,
            attender,
            a_val=x_,
            mask=features['context_num_words'],
            logit_fn='dot',
            tensors=tensors)
        return clue

    x1_clue = get_clue(x1)
    x_start = tf.concat([x1, x1_clue], 2)

    x2 = tf_utils.bi_rnn(
        params.hidden_size,
        x1,
        sequence_length_list=features['context_num_words'],
        scope='bi_rnn_x2',
        training=training,
        dropout_rate=params.dropout_rate)
    x2_clue = tf_utils.bi_rnn(
        params.hidden_size,
        x1_clue,
        sequence_length_list=features['context_num_words'],
        scope='bi_rnn_x2_clue',
        training=training,
        dropout_rate=params.dropout_rate)
    x_end = tf.concat([x2, x2_clue], 2)

    q1 = tf_utils.bi_rnn(
        params.hidden_size,
        q,
        sequence_length_list=features['question_num_words'],
        scope='bi_rnn_q1',
        training=training,
        dropout_rate=params.dropout_rate)

    q2 = tf_utils.bi_rnn(
        params.hidden_size,
        q1,
        sequence_length_list=features['question_num_words'],
        scope='bi_rnn_q2',
        training=training,
        dropout_rate=params.dropout_rate)

    # Self-attention to obtain single vector representation.
    q1_type = tf_utils.self_att(
        q1,
        mask=features['question_num_words'],
        tensors=tensors,
        scope='self_att_q1_type')
    q1_clue = tf_utils.self_att(
        q1,
        mask=features['question_num_words'],
        tensors=tensors,
        scope='self_att_q1_clue')
    q_start = tf.concat([q1_type, q1_clue], 1)
    q2_type = tf_utils.self_att(
        q2, mask=features['question_num_words'], scope='self_att_q2_type')
    q2_clue = tf_utils.self_att(
        q2, mask=features['question_num_words'], scope='self_att_q2_clue')
    q_end = tf.concat([q2_type, q2_clue], 1)

    logits_start, logits_end = _get_logits_from_multihead_x_and_q(
        x_start, x_end, q_start, q_end, features['context_num_words'],
        params.dist)
    return logits_start, logits_end, tensors


def _model_m03(features, mode, params, scope=None):
  """Independent self-attention with LSTM, reaching 60~61%.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope, default is `feature_model`.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """

  with tf.variable_scope(scope or 'feature_model'):
    training = mode == learn.ModeKeys.TRAIN
    tensors = {}

    x, q = embedding_layer(features, mode, params)

    def get_x_and_q(scope=None):
      with tf.variable_scope(scope or 'get_x_and_q'):
        x1 = tf_utils.bi_rnn(
            params.hidden_size,
            x,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_x1',
            training=training,
            dropout_rate=params.dropout_rate)

        attendee = tf_utils.bi_rnn(
            params.hidden_size,
            x1,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attendee',
            training=training,
            dropout_rate=params.dropout_rate)
        attender = tf_utils.bi_rnn(
            params.hidden_size,
            x1,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attender',
            training=training,
            dropout_rate=params.dropout_rate)
        clue = tf_utils.att2d(
            attendee,
            attender,
            a_val=x1,
            mask=features['context_num_words'],
            logit_fn='dot',
            tensors=tensors)

        x_out = tf.concat([x1, clue], 2)

        q1 = tf_utils.bi_rnn(
            params.hidden_size,
            q,
            sequence_length_list=features['question_num_words'],
            scope='bi_rnn_q1',
            training=training,
            dropout_rate=params.dropout_rate)

        q1_type = tf_utils.self_att(
            q1,
            mask=features['question_num_words'],
            tensors=tensors,
            scope='self_att_q1_type')
        q1_clue = tf_utils.self_att(
            q1,
            mask=features['question_num_words'],
            tensors=tensors,
            scope='self_att_q1_clue')
        q_out = tf.concat([q1_type, q1_clue], 1)

        return x_out, q_out

    x_start, q_start = get_x_and_q('start')
    x_end, q_end = get_x_and_q('end')

    logits_start, logits_end = _get_logits_from_multihead_x_and_q(
        x_start, x_end, q_start, q_end, features['context_num_words'],
        params.dist)
    return logits_start, logits_end, tensors


def _model_m04(features, mode, params, scope=None):
  """Regularization with query generation loss on top of m03, reaching 63~64%.

  Note that most part of this model is identical to m03, except for the function
  `reg_gen`, which adds additional generation loss.

  Args:
    features: A dict of feature tensors.
    mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
    params: `params` passed during initialization of `Estimator` object.
    scope: Variable scope, default is `feature_model`.
  Returns:
    `(logits_start, logits_end, tensors)` pair. `tensors` is a dictionary of
    tensors that can be useful outside of this function, e.g. visualization.
  """

  with tf.variable_scope(scope or 'feature_model'):
    training = mode == learn.ModeKeys.TRAIN
    inference = mode == learn.ModeKeys.INFER
    tensors = {}

    with tf.variable_scope('embedding'):
      glove_emb_mat, xv, qv = glove_layer(features)
      _, xc, qc = char_layer(features, params)
      x = tf.concat([xc, xv], 2)
      q = tf.concat([qc, qv], 2)
      x = tf_utils.highway_net(
          x, 2, training=training, dropout_rate=params.dropout_rate)
      q = tf_utils.highway_net(
          q, 2, training=training, dropout_rate=params.dropout_rate, reuse=True)

    def get_x_and_q(scope=None):
      with tf.variable_scope(scope or 'get_x_and_q'):
        x1 = tf_utils.bi_rnn(
            params.hidden_size,
            x,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_x1',
            training=training,
            dropout_rate=params.dropout_rate)

        attendee = tf_utils.bi_rnn(
            params.hidden_size,
            x1,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attendee',
            training=training,
            dropout_rate=params.dropout_rate)
        attender = tf_utils.bi_rnn(
            params.hidden_size,
            x1,
            sequence_length_list=features['context_num_words'],
            scope='bi_rnn_attender',
            training=training,
            dropout_rate=params.dropout_rate)
        clue = tf_utils.att2d(
            attendee,
            attender,
            a_val=x1,
            mask=features['context_num_words'],
            logit_fn='dot',
            tensors=tensors)

        x_out = tf.concat([x1, clue], 2)

        q1 = tf_utils.bi_rnn(
            params.hidden_size,
            q,
            sequence_length_list=features['question_num_words'],
            scope='bi_rnn_q1',
            training=training,
            dropout_rate=params.dropout_rate)

        q1_type = tf_utils.self_att(
            q1,
            mask=features['question_num_words'],
            tensors=tensors,
            scope='self_att_q1_type')
        q1_clue = tf_utils.self_att(
            q1,
            mask=features['question_num_words'],
            tensors=tensors,
            scope='self_att_q1_clue')
        q_out = tf.concat([q1_type, q1_clue], 1)

        return x_out, q_out

    x_start, q_start = get_x_and_q('start')
    x_end, q_end = get_x_and_q('end')

    # TODO(seominjoon): Separate regularization and model parts.
    def reg_gen(glove_emb_mat, memory, scope):
      """Add query generation loss to `losses` collection as regularization."""
      with tf.variable_scope(scope):
        start_vec = tf.get_variable(
            'start_vec', shape=glove_emb_mat.get_shape()[1])
        end_vec = tf.get_variable('end_vec', shape=glove_emb_mat.get_shape()[1])
        glove_emb_mat = tf.concat([
            glove_emb_mat,
            tf.expand_dims(start_vec, 0),
            tf.expand_dims(end_vec, 0)
        ], 0)
        vocab_size = glove_emb_mat.get_shape().as_list()[0]
        start_idx = vocab_size - 2
        end_idx = vocab_size - 1
        batch_size = tf.shape(x)[0]

        # Index memory
        memory_mask = tf.one_hot(
            tf.slice(features['word_answer_%ss' % scope], [0, 0], [-1, 1]),
            tf.shape(x)[1])
        # Transposing below is just a convenient way to do reduction at dim 2
        # and expansion at dim 1 with one operation.
        memory_mask = tf.transpose(memory_mask, [0, 2, 1])
        initial_state = tf.reduce_sum(memory * tf.cast(memory_mask, 'float'), 1)
        cell = tf.contrib.rnn.GRUCell(memory.get_shape().as_list()[-1])

        glove_emb_mat_dense = tf.layers.dense(glove_emb_mat, cell.output_size)

        def deembed(inputs):
          shape = tf.shape(inputs)
          inputs = tf.reshape(inputs, [-1, shape[-1]])
          outputs = tf.matmul(inputs, tf.transpose(glove_emb_mat_dense))
          outputs = tf.reshape(outputs, tf.concat([shape[:-1], [vocab_size]],
                                                  0))
          return outputs

        if inference:
          # During inference, feed previous output to the next input.
          start_tokens = tf.tile(tf.reshape(start_idx, [1]), [batch_size])
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              glove_emb_mat, start_tokens, end_idx)
          helper = tf_utils.DeembedWrapper(helper, deembed)
          maximum_iterations = params.max_gen_length
        else:
          # During training and eval, feed ground truth input all the time.
          q_in = tf_utils.concat_seq_and_tok(qv, start_vec, 'start')
          indexed_q_out = tf_utils.concat_seq_and_tok(
              tf.cast(features['glove_indexed_question_words'], 'int32'),
              end_idx,
              'end',
              sequence_length=features['question_num_words'])
          q_len = tf.cast(features['question_num_words'], 'int32') + 1
          helper = tf.contrib.seq2seq.TrainingHelper(q_in, q_len)
          maximum_iterations = None

        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state)
        (outputs, _), _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)
        logits = deembed(outputs)
        indexed_q_pred = tf.argmax(logits, axis=2, name='indexed_q_pred')
        tensors[indexed_q_pred.op.name] = indexed_q_pred

        if not inference:
          # Add sequence loss to the `losses` collection.
          weights = tf.sequence_mask(q_len, maxlen=tf.shape(indexed_q_out)[1])
          loss = tf.contrib.seq2seq.sequence_loss(logits, indexed_q_out,
                                                  tf.cast(weights, 'float'))
          cf = params.reg_cf * tf.exp(-tf.log(2.0) * tf.cast(
              tf.train.get_global_step(), 'float') / params.reg_half_life)
          tf.add_to_collection('losses', cf * loss)

    if params.reg_gen:
      reg_gen(glove_emb_mat, x_start, 'start')
      reg_gen(glove_emb_mat, x_end, 'end')

    logits_start, logits_end = _get_logits_from_multihead_x_and_q(
        x_start, x_end, q_start, q_end, features['context_num_words'],
        params.dist)
    return logits_start, logits_end, tensors

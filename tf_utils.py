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
"""TensorFlow utilities for extractive question answering models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

VERY_LARGE_NEGATIVE_VALUE = -1e12
VERY_SMALL_POSITIVE_VALUE = 1e-12


def bi_rnn(hidden_size,
           inputs_list,
           sequence_length_list=None,
           scope=None,
           dropout_rate=0.0,
           training=False,
           stack=False,
           cells=None,
           postprocess='concat',
           return_outputs=True,
           out_dim=None,
           reuse=False):
  """Bidirectional RNN with `BasicLSTMCell`.

  Args:
    hidden_size: `int` value, the hidden state size of the LSTM.
    inputs_list: A list of `inputs` tensors, where each `inputs` is
      single sequence tensor with shape [batch_size, seq_len, hidden_size].
      Can be single element instead of list.
    sequence_length_list: A list of `sequence_length` tensors.
      The size of the list should equal to that of `inputs_list`.
      Can be a single element instead of a list.
    scope: `str` value, variable scope for this function.
    dropout_rate: `float` value, dropout rate of LSTM, applied at the inputs.
    training: `bool` value, whether current run is training.
    stack: `bool` value, whether to stack instead of simultaneous bi-LSTM.
    cells: two `RNNCell` instances. If provided, `hidden_size` is ignored.
    postprocess: `str` value: `raw` or `concat` or `add`.
      Postprocessing on forward and backward outputs of LSTM.
    return_outputs: `bool` value, whether to return sequence outputs.
      Otherwise, return the last state.
    out_dim: `bool` value. If `postprocess` is `linear, then this indicates
      the output dim of the linearity.
    reuse: `bool` value, whether to reuse variables.
  Returns:
    A list `return_list` where each element corresponds to each element of
    `input_list`. If the `input_list` is a tensor, also returns a tensor.
  Raises:
    ValueError: If argument `postprocess` is an invalid value.
  """
  if not isinstance(inputs_list, list):
    inputs_list = [inputs_list]
  if sequence_length_list is None:
    sequence_length_list = [None] * len(inputs_list)
  elif not isinstance(sequence_length_list, list):
    sequence_length_list = [sequence_length_list]
  assert len(inputs_list) == len(
      sequence_length_list
  ), '`inputs_list` and `sequence_length_list` must have same lengths.'
  with tf.variable_scope(scope or 'bi_rnn', reuse=reuse) as vs:
    if cells is not None:
      cell_fw = cells[0]
      cell_bw = cells[1]
    else:
      cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=reuse)
      cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=reuse)
    return_list = []
    for inputs, sequence_length in zip(inputs_list, sequence_length_list):
      if return_list:
        vs.reuse_variables()
      if dropout_rate > 0.0:
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
      if stack:
        o_bw, state_bw = tf.nn.dynamic_rnn(
            cell_bw,
            tf.reverse_sequence(inputs, sequence_length, seq_dim=1),
            sequence_length=sequence_length,
            dtype='float',
            scope='rnn_bw')
        o_bw = tf.reverse_sequence(o_bw, sequence_length, seq_dim=1)
        if dropout_rate > 0.0:
          o_bw = tf.layers.dropout(o_bw, rate=dropout_rate, training=training)
        o_fw, state_fw = tf.nn.dynamic_rnn(
            cell_fw,
            o_bw,
            sequence_length=sequence_length,
            dtype='float',
            scope='rnn_fw')
      else:
        (o_fw, o_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            inputs,
            sequence_length=sequence_length,
            dtype='float')
      return_fw = o_fw if return_outputs else state_fw[-1]
      return_bw = o_bw if return_outputs else state_bw[-1]
      if postprocess == 'raw':
        return_ = return_fw, return_bw
      elif postprocess == 'concat':
        return_ = tf.concat([return_fw, return_bw], 2 if return_outputs else 1)
      elif postprocess == 'add':
        return_ = return_fw + return_bw
      elif postprocess == 'max':
        return_ = tf.maximum(return_fw, return_bw)
      elif postprocess == 'linear':
        if out_dim is None:
          out_dim = 2 * hidden_size
        return_ = tf.concat([return_fw, return_bw], 2 if return_outputs else 1)
        return_ = tf.layers.dense(return_, out_dim)
      else:
        return_ = postprocess(return_fw, return_bw)
      return_list.append(return_)
    if len(return_list) == 1:
      return return_list[0]
    return return_list


def exp_mask(logits, mask, mask_is_length=True):
  """Exponential mask for logits.

  Logits cannot be masked with 0 (i.e. multiplying boolean mask)
  because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
  to `False` portion of `mask` so that the portion is effectively ignored
  when exponentiated, e.g. softmaxed.

  Args:
    logits: Arbitrary-rank logits tensor to be masked.
    mask: `boolean` type mask tensor.
      Could be same shape as logits (`mask_is_length=False`)
      or could be length tensor of the logits (`mask_is_length=True`).
    mask_is_length: `bool` value. whether `mask` is boolean mask.
  Returns:
    Masked logits with the same shape of `logits`.
  """
  if mask_is_length:
    mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
  return logits + (1.0 - tf.cast(mask, 'float')) * VERY_LARGE_NEGATIVE_VALUE


def self_att(tensor,
             tensor_val=None,
             mask=None,
             mask_is_length=True,
             logit_fn=None,
             scale_dot=False,
             normalizer=tf.nn.softmax,
             tensors=None,
             scope=None,
             reuse=False):
  """Performs self attention.

  Performs self attention to obtain single vector representation for a sequence
  of vectors.

  Args:
    tensor: [batch_size, sequence_length, hidden_size]-shaped tensor
    tensor_val: If specified, attention is applied on `tensor_val`, i.e.
      `tensor` is key.
    mask: Length mask (shape of [batch_size]) or
      boolean mask ([batch_size, sequence_length])
    mask_is_length: `True` if `mask` is length mask, `False` if it is boolean
      mask
    logit_fn: `logit_fn(tensor)` to obtain logits.
    scale_dot: `bool`, whether to scale the dot product by dividing by
      sqrt(hidden_size).
    normalizer: function to normalize logits.
    tensors: `dict`. If specified, add useful tensors (e.g. attention weights)
      to the `dict` with their (scope) names.
    scope: `string` for defining variable scope
    reuse: Reuse if `True`.
  Returns:
    [batch_size, hidden_size]-shaped tensor.
  """
  assert len(tensor.get_shape()
            ) == 3, 'The rank of `tensor` must be 3 but got {}.'.format(
                len(tensor.get_shape()))
  with tf.variable_scope(scope or 'self_att', reuse=reuse):
    hidden_size = tensor.get_shape().as_list()[-1]
    if logit_fn is None:
      logits = tf.layers.dense(tensor, hidden_size, activation=tf.tanh)
      logits = tf.squeeze(tf.layers.dense(logits, 1), 2)
    else:
      logits = logit_fn(tensor)
    if scale_dot:
      logits /= tf.sqrt(hidden_size)
    if mask is not None:
      logits = exp_mask(logits, mask, mask_is_length=mask_is_length)
    weights = normalizer(logits)
    if tensors is not None:
      weights = tf.identity(weights, name='attention')
      tensors[weights.op.name] = weights
    out = tf.reduce_sum(
        tf.expand_dims(weights, -1) * (tensor
                                       if tensor_val is None else tensor_val),
        1)
    return out


def highway(inputs,
            outputs=None,
            dropout_rate=0.0,
            batch_norm=False,
            training=False,
            scope=None,
            reuse=False):
  """Single-layer highway networks (https://arxiv.org/abs/1505.00387).

  Args:
    inputs: Arbitrary-rank `float` tensor, where the first dim is batch size
      and the last dim is where the highway network is applied.
    outputs: If provided, will replace the perceptron layer (i.e. gating only.)
    dropout_rate: `float` value, input dropout rate.
    batch_norm: `bool` value, whether to use batch normalization.
    training: `bool` value, whether the current run is training.
    scope: `str` value variable scope, default to `highway_net`.
    reuse: `bool` value, whether to reuse variables.
  Returns:
    The output of the highway network, same shape as `inputs`.
  """
  with tf.variable_scope(scope or 'highway', reuse=reuse):
    if dropout_rate > 0.0:
      inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
    dim = inputs.get_shape()[-1]
    if outputs is None:
      outputs = tf.layers.dense(inputs, dim, name='outputs')
      if batch_norm:
        outputs = tf.layers.batch_normalization(outputs, training=training)
      outputs = tf.nn.relu(outputs)
    gate = tf.layers.dense(inputs, dim, activation=tf.nn.sigmoid, name='gate')
    return gate * inputs + (1 - gate) * outputs


def highway_net(inputs,
                num_layers,
                dropout_rate=0.0,
                batch_norm=False,
                training=False,
                scope=None,
                reuse=False):
  """Multi-layer highway networks (https://arxiv.org/abs/1505.00387).

  Args:
    inputs: `float` input tensor to the highway networks.
    num_layers: `int` value, indicating the number of highway layers to build.
    dropout_rate: `float` value for the input dropout rate.
    batch_norm: `bool` value, indicating whether to use batch normalization
      or not.
    training: `bool` value, indicating whether the current run is training
     or not (e.g. eval or inference).
    scope: `str` value, variable scope. Default is `highway_net`.
    reuse: `bool` value, indicating whether the variables in this function
      are reused.
  Returns:
    The output of the highway networks, which is the same shape as `inputs`.
  """
  with tf.variable_scope(scope or 'highway_net', reuse=reuse):
    outputs = inputs
    for i in range(num_layers):
      outputs = highway(
          outputs,
          dropout_rate=dropout_rate,
          batch_norm=batch_norm,
          training=training,
          scope='layer_{}'.format(i))
    return outputs


def char_cnn(inputs,
             out_dim,
             kernel_size,
             dropout_rate=0.0,
             name=None,
             reuse=False,
             activation=None,
             batch_norm=False,
             training=False):
  """Character-level CNN.

  Args:
    inputs: Input tensor of shape [batch_size, num_words, num_chars, in_dim].
    out_dim: `int` value, output dimension of CNN.
    kernel_size: `int` value, the width of the kernel for CNN.
    dropout_rate: `float` value, input dropout rate.
    name: `str` value, variable scope for variables in this function.
    reuse: `bool` value, indicating whether to reuse CNN variables.
    activation: function for activation. Default is `tf.nn.relu`.
    batch_norm: `bool` value, whether to perform batch normalization.
    training: `bool` value, whether the current run is training or not.
  Returns:
    Output tensor of shape [batch_size, num_words, out_dim].

  """
  with tf.variable_scope(name or 'char_cnn', reuse=reuse):
    batch_size = tf.shape(inputs)[0]
    num_words = tf.shape(inputs)[1]
    num_chars = tf.shape(inputs)[2]
    in_dim = inputs.get_shape().as_list()[3]
    if dropout_rate > 0.0:
      inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
    outputs = tf.reshape(
        tf.layers.conv1d(
            tf.reshape(inputs, [batch_size * num_words, num_chars, in_dim]),
            out_dim,
            kernel_size,
            name=name), [batch_size, num_words, -1, out_dim])
    if batch_norm:
      outputs = tf.layers.batch_normalization(outputs, training=training)
    if activation is None:
      activation = tf.nn.relu
    outputs = activation(outputs)
    outputs = tf.reduce_max(outputs, 2)
    return outputs


def att2d(a,
          b,
          a_val=None,
          mask=None,
          b_mask=None,
          a_null=None,
          logit_fn='bahdanau',
          return_weights=False,
          normalizer=tf.nn.softmax,
          transpose=False,
          scale_logits=False,
          reduce_fn=None,
          tensors=None,
          scope=None):
  """2D-attention on a pair of sequences.

  Obtain the most similar (most attended) vector among the vectors in `a` to
  each vector in `b`. That is, `b` can be considered as key and `a` is value.
  Or in other words, `b` is attender and `a` is attendee.

  Args:
    a: [batch_size, a_len, hidden_size] shaped tensor.
    b: [batch_size, b_len, hidden_size] shaped tensor.
    a_val: If specified, attention is performed on `a_val` instead of `a`.
    mask: length mask tensor, boolean mask tensor for `a`, or 2d mask of size
      [b_len, a_len].
    b_mask: something.
    a_null: If specified, this becomes a possible vector to be attended.
    logit_fn: `logit_fn(a, b)` computes logits for attention. By default,
      uses attention function by Bahdanau et al (2014). Can be string `dot`,
      in which case the dot product is memory-efficiently computed via
      `tf.batch_matmul`.
    return_weights: `bool` value, whether to return weights instead of
      the attended vector.
    normalizer: function that normalizes the weights.
    transpose: `bool`, whether to transpose the normalizer axis.
    scale_logits: `bool`, whether to scale the logits by
      sqrt(hidden_size), as shown in https://arxiv.org/abs/1706.03762.
    reduce_fn: python fn, which reduces the logit matrix in the b's axis
      if specified.
    tensors: `dict`. If specified, add useful tensors (e.g. attention weights)
      to the `dict` with their (scope) names.
    scope: `str` value, indicating the variable scope of this function.
  Returns:
    [batch_size, b_len, hidden_size] shaped tensor, where each vector
    represents the most similar vector among `a` for each vector in `b`.
    If `return_weights` is `True`, return tensor is
    [batch_size, b_len, a_len] shape. If `reduce_fn` is specified, the return
    tensor shape is [batch_size, 1, a_len]
  """
  with tf.variable_scope(scope or 'att_2d'):
    batch_size = tf.shape(a)[0]
    hidden_size = a.get_shape().as_list()[-1]
    if a_null is not None:
      a_null = tf.tile(
          tf.expand_dims(tf.expand_dims(a_null, 0), 0), [batch_size, 1, 1])
      a = tf.concat([a_null, a], 1)
      if mask is not None:
        # TODO(seominjoon) : To support other shapes of masks.
        assert len(mask.get_shape()) == 1
        mask += 1

    # Memory-efficient operation for dot product logit function.
    if logit_fn == 'wdot':
      weights = tf.get_variable('weights', shape=[hidden_size], dtype='float')
      bias = tf.get_variable('bias', shape=[], dtype='float')
      logits = tf.matmul(b * weights, a, transpose_b=True) + bias
    elif logit_fn == 'dot':
      logits = tf.matmul(b, a, transpose_b=True)
    elif logit_fn == 'bilinear':
      logits = tf.matmul(
          tf.layers.dense(b, hidden_size, use_bias=False), a, transpose_b=True)
    elif logit_fn == 'l2':
      ba = tf.matmul(b, a, transpose_b=True)
      aa = tf.expand_dims(tf.reduce_sum(a * a, 2), 1)
      bb = tf.expand_dims(tf.reduce_sum(b * b, 2), 2)
      logits = 2 * ba - aa - bb
    else:
      # WARNING : This is memory-intensive!
      aa = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[1], 1, 1])
      bb = tf.tile(tf.expand_dims(b, 2), [1, 1, tf.shape(a)[1], 1])
      if logit_fn == 'bahdanau':
        logits = tf.layers.dense(
            tf.concat([aa, bb], 3), hidden_size, activation=tf.tanh)
        logits = tf.squeeze(tf.layers.dense(logits, 1), 3)
      else:
        logits = logit_fn(aa, bb)  # [batch_size, a_len, b_len]-shaped tensor.
    if scale_logits:
      logits /= tf.sqrt(tf.cast(hidden_size, 'float'))
    if mask is not None:
      if len(mask.get_shape()) == 1:
        mask = tf.sequence_mask(mask, tf.shape(a)[1])
      if len(mask.get_shape()) == 2:
        mask = tf.expand_dims(mask, 1)
        if b_mask is not None:
          if len(b_mask.get_shape()) == 1:
            b_mask = tf.sequence_mask(b_mask, tf.shape(b)[1])
          mask &= tf.expand_dims(b_mask, -1)
      logits = exp_mask(logits, mask, mask_is_length=False)
    if reduce_fn:
      logits = tf.expand_dims(reduce_fn(logits, 1), 1)
    if transpose:
      logits = tf.transpose(logits, [0, 2, 1])
    p = logits if normalizer is None else normalizer(logits)
    if transpose:
      p = tf.transpose(p, [0, 2, 1])
      logits = tf.transpose(logits, [0, 2, 1])
    if tensors is not None:
      p = tf.identity(p, name='attention')
      tensors[p.op.name] = p

    if return_weights:
      return p

    # Memory-efficient application of attention weights.
    # [batch_size, b_len, hidden_size]
    a_b = tf.matmul(p, a if a_val is None else a_val)
    return a_b


def mlp(a,
        hidden_sizes,
        activation=tf.nn.relu,
        activate_last=True,
        dropout_rate=0.0,
        training=False,
        scope=None):
  """Multi-layer perceptron.

  Args:
    a: input tensor.
    hidden_sizes: `list` of `int`, hidden state sizes for perceptron layers.
    activation: function handler for activation.
    activate_last: `bool`, whether to activate the last layer or not.
    dropout_rate: `float`, dropout rate at the input of each layer.
    training: `bool`, whether the current run is training or not.
    scope: `str`, variable scope of all tensors and weights in this function.
  Returns:
    Tensor with same shape as `a` except for the last dim, whose size is equal
    to `hidden_sizes[-1]`.
  """
  with tf.variable_scope(scope or 'mlp'):
    for idx, hidden_size in enumerate(hidden_sizes):
      with tf.variable_scope('layer_%d' % idx):
        if dropout_rate > 0.0:
          a = tf.layers.dropout(a, rate=dropout_rate, training=training)
        activate = idx < len(hidden_sizes) - 1 or activate_last
        a = tf.layers.dense(
            a, hidden_size, activation=activation if activate else None)

    return a


def split_concat(a, b, num, axis=None):
  if axis is None:
    axis = len(a.get_shape()) - 1
  a_list = tf.split(a, num, axis=axis)
  b_list = tf.split(b, num, axis=axis)
  t_list = tuple(
      tf.concat([aa, bb], axis=axis) for aa, bb in zip(a_list, b_list))
  return t_list


def concat_seq_and_tok(sequence, token, position, sequence_length=None):
  """Concatenates a token to the given sequence, either at the start or end.

  The token's dimension should match the last dimension of the sequence.

  Args:
    sequence: [batch_size, sequence_length] shaped tensor or
      [batch_size, sequence_length, hidden_size] shaped tensor.
    token: scalar tensor or [hidden_size] shaped tensor.
    position: `str`, either 'start' or 'end'.
    sequence_length: [batch_size] shaped `int64` tensor. Must be specified
      if `position` is 'end'.
  Returns:
    [batch_size, sequence_length+1] or
    [batch_size, sequence_length+1, hidden_size] shaped tensor.
  Raises:
    ValueError: If `position` is not 'start' or 'end'.
  """
  batch_size = tf.shape(sequence)[0]
  if len(sequence.get_shape()) == 3:
    token = tf.tile(
        tf.expand_dims(tf.expand_dims(token, 0), 0), [batch_size, 1, 1])
  elif len(sequence.get_shape()) == 2:
    token = tf.tile(tf.reshape(token, [1, 1]), [batch_size, 1])

  if position == 'start':
    sequence = tf.concat([token, sequence], 1)
  elif position == 'end':
    assert sequence_length is not None
    sequence = tf.reverse_sequence(sequence, sequence_length, seq_axis=1)
    sequence = tf.concat([token, sequence], 1)
    sequence = tf.reverse_sequence(sequence, sequence_length + 1, seq_axis=1)
  else:
    raise ValueError('%r is an invalid argument for `position`.' % position)
  return sequence


class ExternalInputWrapper(tf.contrib.rnn.RNNCell):
  """Wrapper for `RNNCell`, concatenates an external tensor to the input."""

  def __init__(self, cell, external_input, reuse=False):
    super(ExternalInputWrapper, self).__init__(_reuse=reuse)
    self._cell = cell
    self._external = external_input

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def call(self, inputs, state):
    inputs = tf.concat([self._external_input, inputs], 1)
    return self._cell(inputs, state)


class DeembedWrapper(tf.contrib.seq2seq.Helper):
  """Wrapper for `Helper`, applies given deembed function to the output.

  The deembed function has single input and single output.
  It is applied on the output of the previous RNN before feeding it into the
  next time step.
  """

  def __init__(self, helper, deembed_fn):
    self._helper = helper
    self._deembed_fn = deembed_fn

  @property
  def batch_size(self):
    return self._helper.batch_size

  def initialize(self, name=None):
    return self._helper.initialize(name=name)

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    outputs = self._deembed_fn(outputs)
    return self._helper.next_inputs(time, outputs, state, sample_ids, name=name)

  def sample(self, time, outputs, state, name=None):
    outputs = self._deembed_fn(outputs)
    return self._helper.sample(time, outputs, state, name=name)

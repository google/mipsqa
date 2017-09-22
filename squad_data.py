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
"""SQuAD data parsing module for tf.learn model.

This module loads TFRecord and hyperparameters from a specified directory
(files dumped by `squad_prepro.py`) and provides tensors for data feeding.
This module also provides data-specific functions for evaluation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import json
import os
import re
import string

import numpy as np
import tensorflow as tf

import squad_prepro


def get_input_fn(root_data_dir,
                 glove_dir,
                 data_type,
                 batch_size,
                 glove_size,
                 shuffle_files=True,
                 shuffle_examples=False,
                 queue_capacity=5000,
                 min_after_dequeue=1000,
                 num_epochs=None,
                 oom_test=False):
  """Get input function for the given data type from the given data directory.

  Args:
    root_data_dir: The directory to load data from. Corresponds to `to_dir`
      of `squad_prepro_main.py` file.
    glove_dir: path to the directory that contains GloVe files.
    data_type: `str` object, either `train` or `dev`.
    batch_size: Batch size of the inputs.
    glove_size: size of GloVe vector to load.
    shuffle_files: If `True`, shuffle the queue for the input files.
    shuffle_examples: If `True`, shuffle the queue for the examples.
    queue_capacity: `int`, maximum number of examples in input queue.
    min_after_dequeue: `int`, for`RandomShuffleQueue`, minimum number of
      examples before dequeueing to ensure randomness.
    num_epochs: Number of epochs on the data. `None` means infinite.
      This queue comes after the file queue.
    oom_test: Stress test to see if the current dataset and model causes
      out-of-memory error on GPU.
  Returns:
    Function definition `input_fn` compatible with `Experiment` object.
  """
  filenames = tf.gfile.Glob(
      os.path.join(root_data_dir, data_type, 'data', 'squad_data_*'))
  tf.logging.info('reading examples from following files:')
  for filename in filenames:
    tf.logging.info(filename)
  sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.int64, allow_missing=True)
  str_sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.string, allow_missing=True)
  int_feature = tf.FixedLenFeature([], tf.int64)
  str_feature = tf.FixedLenFeature([], tf.string)
  # Let N = batch_size, JX = max num context words, JQ = max num ques words,
  # C = num chars per word (fixed, default = 16)
  features = {
      'indexed_context_words': sequence_feature,  # Shape = [JX]
      'glove_indexed_context_words': sequence_feature,
      'indexed_context_chars': sequence_feature,  # Shape = [JX * C]
      'indexed_question_words': sequence_feature,  # Shape = [JQ]
      'glove_indexed_question_words': sequence_feature,
      'indexed_question_chars': sequence_feature,  # Shape = [JQ * C]
      'word_answer_starts': sequence_feature,  # Answer start index.
      'word_answer_ends': sequence_feature,  # Answer end index.
      'context_num_words':
          int_feature,  # Number of context words in each example. [A]
      'question_num_words':
          int_feature,  # Number of question words in each example. [A]
      'answers': str_sequence_feature,  # List of answers in each example. [A]
      'context_words': str_sequence_feature,  # [JX]
      'question_words': str_sequence_feature,  # [JQ]
      'context': str_feature,
      'id': str_feature,
      'num_answers': int_feature,
      'question': str_feature,
  }

  exp_metadata_path = os.path.join(root_data_dir, 'metadata.json')
  with tf.gfile.GFile(exp_metadata_path, 'r') as fp:
    exp_metadata = json.load(fp)

  metadata_path = os.path.join(root_data_dir, data_type, 'metadata.json')
  with tf.gfile.GFile(metadata_path, 'r') as fp:
    metadata = json.load(fp)
  emb_mat = squad_prepro.get_idx2vec_mat(glove_dir, glove_size,
                                         metadata['glove_word2idx'])

  def _input_fn():
    """Input function compatible with `Experiment` object.

    Returns:
      A tuple of feature tensors and target tensors.
    """
    # TODO(seominjoon): There is bottleneck in data feeding, slow for N >= 128.
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=shuffle_files, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, se = reader.read(filename_queue)
    # TODO(seominjoon): Consider moving data filtering to here.
    features_op = tf.parse_single_example(se, features=features)

    names = list(features_op.keys())
    dtypes = [features_op[name].dtype for name in names]
    shapes = [features_op[name].shape for name in names]

    if shuffle_examples:
      # Data shuffling.
      rq = tf.RandomShuffleQueue(
          queue_capacity, min_after_dequeue, dtypes, names=names)
    else:
      rq = tf.FIFOQueue(queue_capacity, dtypes, names=names)
    enqueue_op = rq.enqueue(features_op)
    dequeue_op = rq.dequeue()
    dequeue_op = [dequeue_op[name] for name in names]
    qr = tf.train.QueueRunner(rq, [enqueue_op])
    tf.train.add_queue_runner(qr)

    batch = tf.train.batch(
        dequeue_op,
        batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        shapes=shapes,
        allow_smaller_final_batch=True,
        num_threads=5)
    batch = {name: each for name, each in zip(names, batch)}
    target_keys = [
        'word_answer_starts', 'word_answer_ends', 'answers', 'num_answers'
    ]
    # TODO(seominjoon) For cheating-safe, comment out #.
    features_batch = {
        key: val
        for key, val in batch.items()  # if key not in target_keys
    }

    # `metadata['emb_mat`]` contains GloVe embedding, and `xv` in
    # `features_batch` index into the vectors.
    features_batch['emb_mat'] = tf.constant(emb_mat)
    targets_batch = {key: batch[key] for key in target_keys}

    # Postprocessing for character data.
    # Due to the limitation of the python wrapper for prototxt,
    # the characters (by index) need to be flattened when saving on prototxt.
    # The following 'unflattens' the character tensor.
    actual_batch_size = tf.shape(batch['indexed_context_chars'])[0]
    features_batch['indexed_context_chars'] = tf.reshape(
        features_batch['indexed_context_chars'],
        [actual_batch_size, -1, metadata['num_chars_per_word']])
    features_batch['indexed_question_chars'] = tf.reshape(
        features_batch['indexed_question_chars'],
        [actual_batch_size, -1, metadata['num_chars_per_word']])

    # Make sure answer start and end positions are less than sequence lengths.
    # TODO(seominjoon) This will need to move to a separate test.
    with tf.control_dependencies([
        tf.assert_less(
            tf.reduce_max(targets_batch['word_answer_starts'], 1),
            features_batch['context_num_words'])
    ]):
      targets_batch['word_answer_starts'] = tf.identity(
          targets_batch['word_answer_starts'])
    with tf.control_dependencies([
        tf.assert_less(
            tf.reduce_max(targets_batch['word_answer_ends'], 1),
            features_batch['context_num_words'])
    ]):
      targets_batch['word_answer_ends'] = tf.identity(
          targets_batch['word_answer_ends'])

    # Stress test to ensure no OOM for GPU occurs.
    if oom_test:
      features_batch['indexed_context_words'] = tf.constant(
          np.ones(
              [batch_size, exp_metadata['max_context_size']], dtype='int64'))
      features_batch['glove_indexed_context_words'] = tf.constant(
          np.ones(
              [batch_size, exp_metadata['max_context_size']], dtype='int64'))
      features_batch['indexed_context_chars'] = tf.constant(
          np.ones(
              [
                  batch_size, exp_metadata['max_context_size'], exp_metadata[
                      'num_chars_per_word']
              ],
              dtype='int64'))
      features_batch['indexed_question_words'] = tf.constant(
          np.ones([batch_size, exp_metadata['max_ques_size']], dtype='int64'))
      features_batch['glove_indexed_question_words'] = tf.constant(
          np.ones([batch_size, exp_metadata['max_ques_size']], dtype='int64'))
      features_batch['indexed_question_chars'] = tf.constant(
          np.ones(
              [
                  batch_size, exp_metadata['max_ques_size'], exp_metadata[
                      'num_chars_per_word']
              ],
              dtype='int64'))
      features_batch['question_num_words'] = tf.constant(
          np.ones([batch_size], dtype='int64') * exp_metadata['max_ques_size'])
      features_batch['context_num_words'] = tf.constant(
          np.ones([batch_size], dtype='int64') *
          exp_metadata['max_context_size'])

    return features_batch, targets_batch

  return _input_fn


def get_params(root_data_dir):
  """Load data-specific parameters from `root_data_dir`.

  Args:
    root_data_dir: The data directory to load parameter files from.
    This is equivalent to the `output_dir` of `data/squad_prepro.py`.
  Returns:
    A dict of hyperparameters.
  """
  indexer_path = os.path.join(root_data_dir, 'indexer.json')
  with tf.gfile.GFile(indexer_path, 'r') as fp:
    indexer = json.load(fp)

  return {
      'vocab_size': len(indexer['word2idx']),
      'char_vocab_size': len(indexer['char2idx']),
  }


def get_eval_metric_ops(targets, predictions):
  """Get a dictionary of eval metrics for `Experiment` object.

  Args:
    targets: `targets` that go into `model_fn` of `Experiment`.
    predictions: Dictionary of predictions, output of `get_preds`.
  Returns:
    A dictionary of eval metrics.
  """
  # TODO(seominjoon): yp should also consider no answer case.
  yp1 = tf.expand_dims(predictions['yp1'], -1)
  yp2 = tf.expand_dims(predictions['yp2'], -1)
  answer_mask = tf.sequence_mask(targets['num_answers'])
  start_correct = tf.reduce_any(
      tf.equal(targets['word_answer_starts'], yp1) & answer_mask, 1)
  end_correct = tf.reduce_any(
      tf.equal(targets['word_answer_ends'], yp2) & answer_mask, 1)
  correct = start_correct & end_correct
  em = tf.py_func(
      _enum_fn(_exact_match_score, dtype='float32'), [
          predictions['a'], targets['answers'], predictions['has_answer'],
          answer_mask
      ], 'float32')
  f1 = tf.py_func(
      _enum_fn(_f1_score, dtype='float32'), [
          predictions['a'], targets['answers'], predictions['has_answer'],
          answer_mask
      ], 'float32')

  eval_metric_ops = {
      'acc1': tf.metrics.mean(tf.cast(start_correct, 'float')),
      'acc2': tf.metrics.mean(tf.cast(end_correct, 'float')),
      'acc': tf.metrics.mean(tf.cast(correct, 'float')),
      'em': tf.metrics.mean(em),
      'f1': tf.metrics.mean(f1),
  }
  return eval_metric_ops


def get_answer_op(context, context_words, answer_start, answer_end):
  return tf.py_func(
      _enum_fn(_get_answer), [context, context_words, answer_start, answer_end],
      'string')


def _get_answer(context, context_words, answer_start, answer_end):
  """Get answer given context, context_words, and span.

  Args:
    context: A list of bytes, to be decoded with utf-8.
    context_words: A list of a list of bytes, to be decoded with utf-8.
    answer_start: An int for answer start.
    answer_end: An int for answer end.
  Returns:
    A list of bytes, encoded with utf-8, for the answer.
  """
  context = context.decode('utf-8')
  context_words = [word.decode('utf-8') for word in context_words]
  pos = 0
  answer_start_char = None
  answer_end_char = None
  for i, word in enumerate(context_words):
    pos = context.index(word, pos)
    if answer_start == i:
      answer_start_char = pos
    pos += len(word)
    if answer_end == i:
      answer_end_char = pos
      break
  assert answer_start_char is not None, (
      '`answer_start` is not found in context. '
      'context=`%s`, context_words=`%r`, '
      'answer_start=%d, answer_end=%d') % (context, context_words, answer_start,
                                           answer_end)
  assert answer_end_char is not None, (
      '`answer_end` is not found in context. '
      'context=`%s`, context_words=`%r`, '
      'answer_start=%d, answer_end=%d') % (context, context_words, answer_start,
                                           answer_end)
  answer = context[answer_start_char:answer_end_char].encode('utf-8')
  return answer


def _f1_score(prediction, ground_truths, has_answer, answer_mask):
  prediction = prediction.decode('utf-8')
  ground_truths = [
      ground_truth.decode('utf-8') for ground_truth in ground_truths
  ]
  if not has_answer:
    return float(ground_truths[0] == squad_prepro.NO_ANSWER)
  elif ground_truths[0] == squad_prepro.NO_ANSWER:
    return 0.0
  else:
    scores = np.array([
        _f1_score_(prediction, ground_truth) for ground_truth in ground_truths
    ])
    return max(scores * answer_mask.astype(float))


def _exact_match_score(prediction, ground_truths, has_answer, answer_mask):
  prediction = prediction.decode('utf-8')
  ground_truths = [
      ground_truth.decode('utf-8') for ground_truth in ground_truths
  ]
  if not has_answer:
    return float(ground_truths[0] == squad_prepro.NO_ANSWER)
  elif ground_truths[0] == squad_prepro.NO_ANSWER:
    return 0.0
  else:
    scores = np.array([
        float(_exact_match_score_(prediction, ground_truth))
        for ground_truth in ground_truths
    ])
    return max(scores * answer_mask.astype(float))


def _enum_fn(fn, dtype='object'):

  def new_fn(*args):
    return np.array([fn(*each_args) for each_args in zip(*args)], dtype=dtype)

  return new_fn


# Functions below are copied from official SQuAD eval script and SHOULD NOT
# BE MODIFIED.


def _normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace.

  Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED.

  Args:
    s: Input text.
  Returns:
    Normalized text.
  """

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score_(prediction, ground_truth):
  """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
  prediction_tokens = _normalize_answer(prediction).split()
  ground_truth_tokens = _normalize_answer(ground_truth).split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def _exact_match_score_(prediction, ground_truth):
  """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
  return _normalize_answer(prediction) == _normalize_answer(ground_truth)

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
"""Preprocesser for SQuAD.

`from_dir` will need to contain original SQuAD train/dev data:

- `train-v1.1.json`
- `dev-v1.1.json`

In each TFRecord file, following features will be provided. Note that all
strings are encoded with utf-8.

Directly from the data.
- `id` : string id from original SQuAD data.
- `version` : the version of SQuAD data.
- `title` : the title of the article.
- `question` : original question string.
- `context` : original context string.

- `answers` : original list of answer strings. Variable length.
- `answer_starts` : original list of integers for answer starts.

Processed.
- `question_words` : tokenized question. A variable len list of strings.
- `context_words` : tokenized context. Variable length.
- `question_chars` : question chars.
- `context_chars` : context chars.
- `indexed_question_words` : question words indexed by vocab.
- `indexed_context_words` : context words indexed by vocab.
- `glove_indexed_question_words` : question words indexed by GloVe.
- `glove_indexed_context_words` : context words indexed by GloVe.
- `indexed_question_chars` : question chars, flattened and indexed.
- `indexed_context_chars` : ditto.
- `question_num_words` : number of words in question.
- `context_num_words` : number of words in context.
- `is_supervised` : whether the dataset is supervised or not.

- `num_answers': integer indicating number of answers. 0 means no answer.
- `word_answer_starts` : word-level answer start positions.
- `word_answer_ends` : word-level answer end positions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import os
import random
from six import string_types

import tensorflow as tf
from tqdm import tqdm
import squad_prepro

tf.flags.DEFINE_string('from_dir', '', 'Directory for original SQuAD data.')
tf.flags.DEFINE_string('to_dir', '', 'Directory for preprocessed SQuAD data.')
tf.flags.DEFINE_string('glove_dir', '', 'Directory for GloVe files.')
tf.flags.DEFINE_string('indexer_dir', '', 'Directory for indexer. '
                       'If specified, does not load train and uses this.')
tf.flags.DEFINE_integer('word_count_th', 100, 'Word count threshold for vocab.')
tf.flags.DEFINE_integer('char_count_th', 100, 'Char count threshold for vocab.')
tf.flags.DEFINE_integer('max_context_size', 256,
                        'Maximum context size. Set this to `0` for test, which '
                        'sets no limit to the maximum context size.')
tf.flags.DEFINE_integer('max_ques_size', 32,
                        'Maximum question size. Set this to `0` for test.')
tf.flags.DEFINE_integer('num_chars_per_word', 16,
                        'Fixed number of characters per word.')
tf.flags.DEFINE_boolean(
    'split', False,
    'if `True`, each context will be sentence instead of paragraph, '
    'and answer label (word index) will be `None` in case of no answer.')
tf.flags.DEFINE_boolean('filter', False,
                        'If `True`, filters data by context and question sizes')
tf.flags.DEFINE_boolean('sort', False,
                        'If `True`, sorts data by context length.')
tf.flags.DEFINE_boolean('shuffle', True, 'If `True`, shuffle examples.')
tf.flags.DEFINE_boolean('draft', False, 'If `True`, fast draft mode, '
                        'which only loads first few examples.')
tf.flags.DEFINE_integer('max_shard_size', 1000, 'Max size of each shard.')
tf.flags.DEFINE_integer('positive_augment_factor', 0, 'Augment positive '
                        'examples by this factor.')
tf.flags.DEFINE_boolean('answerable', False, 'This flag '
                        'allows one to use only examples that have answers.')

FLAGS = tf.flags.FLAGS


def get_tf_example(example):
  """Get `tf.train.Example` object from example dict.

  Args:
    example: tokenized, indexed example.
  Returns:
    `tf.train.Example` object corresponding to the example.
  Raises:
    ValueError: if a key in `example` is invalid.
  """
  feature = {}
  for key, val in example.items():
    if not isinstance(val, list):
      val = [val]
    if val:
      if isinstance(val[0], string_types):
        dtype = 'bytes'
      elif isinstance(val[0], int):
        dtype = 'int64'
      else:
        raise TypeError('`%s` has an invalid type: %r' % (key, type(val[0])))
    else:
      if key == 'answers':
        dtype = 'bytes'
      elif key in ['answer_starts', 'word_answer_starts', 'word_answer_ends']:
        dtype = 'int64'
      else:
        raise ValueError(key)

    if dtype == 'bytes':
      # Transform unicode into bytes if necessary.
      val = [each.encode('utf-8') for each in val]
      feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=val))
    elif dtype == 'int64':
      feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    else:
      raise TypeError('`%s` has an invalid type: %r' % (key, type(val[0])))
  return tf.train.Example(features=tf.train.Features(feature=feature))


def dump(examples, metadata, data_type):
  """Dumps examples as TFRecord files.

  Args:
    examples: a `list` of `dict`, where each dict is indexed example.
    metadata: `dict`, metadata corresponding to the examples.
    data_type: `str`, representing the data type of the examples (e.g. `train`).
  """
  out_dir = os.path.join(FLAGS.to_dir, data_type)
  metadata_path = os.path.join(out_dir, 'metadata.json')
  data_dir = os.path.join(out_dir, 'data')
  tf.gfile.MakeDirs(out_dir)
  tf.gfile.MakeDirs(data_dir)

  with tf.gfile.GFile(metadata_path, 'w') as fp:
    json.dump(metadata, fp)

  # Dump stuff
  writer = None
  counter = 0
  num_shards = 0
  for example in tqdm(examples):
    if writer is None:
      path = os.path.join(data_dir,
                          'squad_data_{}'.format(str(num_shards).zfill(4)))
      writer = tf.python_io.TFRecordWriter(path)
    tf_example = get_tf_example(example)
    writer.write(tf_example.SerializeToString())
    counter += 1
    if counter == FLAGS.max_shard_size:
      counter = 0
      writer.close()
      writer = None
      num_shards += 1
  if writer is not None:
    writer.close()


def prepro(data_type, indexer=None):
  """Preprocesses the given data type."""
  squad_path = os.path.join(FLAGS.from_dir, '%s-v1.1.json' % data_type)
  tf.logging.info('Loading %s' % squad_path)
  examples = squad_prepro.get_examples(squad_path)

  if FLAGS.draft:
    examples = random.sample(examples, 100)

  if FLAGS.split:
    tf.logging.info('Splitting each example')
    tf.logging.info('Before splitting: %d %s examples' % (len(examples),
                                                          data_type))
    examples = list(
        itertools.chain(* [
            squad_prepro.split(
                e, positive_augment_factor=FLAGS.positive_augment_factor)
            for e in tqdm(examples)
        ]))
    tf.logging.info('After splitting: %d %s examples' % (len(examples),
                                                         data_type))

  if FLAGS.answerable:
    tf.logging.info('Using only answerable examples.')
    examples = [
        example for example in examples
        if example['answers'][0] != squad_prepro.NO_ANSWER
    ]

  if FLAGS.shuffle:
    tf.logging.info('Shuffling examples')
    random.shuffle(examples)

  if indexer is None:
    tf.logging.info('Creating indexer')
    indexer = squad_prepro.SquadIndexer(FLAGS.glove_dir, draft=FLAGS.draft)

    tf.logging.info('Indexing %s data' % data_type)
    indexed_examples, metadata = indexer.fit(
        examples,
        min_word_count=FLAGS.word_count_th,
        min_char_count=FLAGS.char_count_th,
        num_chars_per_word=FLAGS.num_chars_per_word)
  else:
    indexed_examples, metadata = indexer.prepro_eval(
        examples, num_chars_per_word=FLAGS.num_chars_per_word)
  tf.gfile.MakeDirs(FLAGS.to_dir)
  indexer_save_path = os.path.join(FLAGS.to_dir, 'indexer.json')
  tf.logging.info('Saving indexer')
  indexer.save(indexer_save_path)

  if FLAGS.filter:
    tf.logging.info('Filtering examples')
    tf.logging.info('Before filtering: %d %s examples' % (len(indexed_examples),
                                                          data_type))
    indexed_examples = [
        e for e in indexed_examples
        if len(e['context_words']) <= FLAGS.max_context_size and
        len(e['question_words']) <= FLAGS.max_ques_size
    ]
    tf.logging.info('After filtering: %d %s examples' % (len(indexed_examples),
                                                         data_type))
    tf.logging.info('Has answers: %d %s examples' %
                    (sum(1 for e in indexed_examples
                         if e['answer_starts'][0] >= 0), data_type))
    metadata['max_context_size'] = max(
        len(e['context_words']) for e in indexed_examples)
    metadata['max_ques_size'] = max(
        len(e['question_words']) for e in indexed_examples)

  if FLAGS.sort:
    tf.logging.info('Sorting examples')
    indexed_examples = sorted(
        indexed_examples, key=lambda e: len(e['context_words']))

  tf.logging.info('Dumping %s examples' % data_type)
  dump(indexed_examples, metadata, data_type)

  return indexer, metadata


def main(argv):
  del argv
  assert not tf.gfile.Exists(FLAGS.to_dir), '%s already exists.' % FLAGS.to_dir

  if FLAGS.indexer_dir:
    indexer_path = os.path.join(FLAGS.indexer_dir, 'indexer.json')
    tf.logging.info('Loading indexer from %s' % indexer_path)
    indexer = squad_prepro.SquadIndexer(FLAGS.glove_dir, draft=FLAGS.draft)
    indexer.load(indexer_path)
  else:
    indexer, train_metadata = prepro('train')
  _, dev_metadata = prepro('dev', indexer=indexer)

  if FLAGS.indexer_dir:
    exp_metadata = dev_metadata
  else:
    exp_metadata = {
        'max_context_size':
            max(train_metadata['max_context_size'],
                dev_metadata['max_context_size']),
        'max_ques_size':
            max(train_metadata['max_ques_size'], dev_metadata['max_ques_size']),
        'num_chars_per_word':
            max(train_metadata['num_chars_per_word'],
                dev_metadata['num_chars_per_word'])
    }

  tf.logging.info('Dumping experiment metadata')
  exp_metadata_path = os.path.join(FLAGS.to_dir, 'metadata.json')
  with tf.gfile.GFile(exp_metadata_path, 'w') as fp:
    json.dump(exp_metadata, fp)


if __name__ == '__main__':
  tf.app.run(main)

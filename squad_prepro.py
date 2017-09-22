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
"""Library for preprocessing SQuAD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
import itertools
import json
import os

import nltk
import numpy as np
import tensorflow as tf
from tqdm import tqdm

PAD = u'<PAD>'
UNK = u'<UNK>'
NO_ANSWER = u'<NO_ANSWER>'


class SquadIndexer(object):
  """Indexer for SQuAD.

  Instantiating this class loads GloVe. The object can fit examples (creating
  vocab out of the examples) and index the examples.
  """

  def __init__(self, glove_dir, tokenizer=None, draft=False):
    self._glove_words = list(_get_glove(glove_dir, size=50, draft=draft))
    self._glove_vocab = _get_glove_vocab(self._glove_words)
    self._glove_dir = glove_dir
    self._draft = draft
    self._tokenizer = tokenizer or _word_tokenize
    self._word2idx_dict = None
    self._char2idx_dict = None

  def word2idx(self, word):
    word = word.lower()
    if word not in self._word2idx_dict:
      return 1
    return self._word2idx_dict[word]

  def char2idx(self, char):
    if char not in self._char2idx_dict:
      return 1
    return self._char2idx_dict[char]

  def fit(self,
          examples,
          min_word_count=10,
          min_char_count=100,
          num_chars_per_word=16):
    """Fits examples and return indexed examples with metadata.

    Fitting examples means the vocab is created out of the examples.
    The vocab can be saved via `save` and loaded via `load` methods.

    Args:
      examples: list of dictionary, where each dictionary is an example.
      min_word_count: `int` value, minimum word count to be included in vocab.
      min_char_count: `int` value, minimum char count to be included in vocab.
      num_chars_per_word: `int` value, number of chars to store per word.
        This is fixed, so if word is shorter, then the rest is padded with 0.
        The characters are flattened, so need to be reshaped when using them.
    Returns:
      a tuple `(indexed_examples, metadata)`, where `indexed_examples` is a
      list of dict (each dict being indexed example) and `metadata` is a dict
      of `glove_word2idx_dict` and statistics of the examples.
    """
    tokenized_examples = [
        _tokenize(example, self._tokenizer, num_chars_per_word)
        for example in tqdm(examples, 'tokenizing')
    ]
    word_counter = _get_word_counter(tokenized_examples)
    char_counter = _get_char_counter(tokenized_examples)
    self._word2idx_dict = _counter2vocab(word_counter, min_word_count)
    tf.logging.info('Word vocab size: %d' % len(self._word2idx_dict))
    self._char2idx_dict = _counter2vocab(char_counter, min_char_count)
    tf.logging.info('Char vocab size: %d' % len(self._char2idx_dict))
    glove_word2idx_dict = _get_glove_vocab(
        self._glove_words, counter=word_counter)
    tf.logging.info('Glove word vocab size: %d' % len(glove_word2idx_dict))

    def glove_word2idx(word):
      word = word.lower()
      return glove_word2idx_dict[word] if word in glove_word2idx_dict else 1

    indexed_examples = [
        _index(example, self.word2idx, glove_word2idx, self.char2idx)
        for example in tqdm(tokenized_examples, desc='indexing')
    ]

    metadata = self._get_metadata(indexed_examples)
    metadata['glove_word2idx'] = glove_word2idx_dict
    metadata['num_chars_per_word'] = num_chars_per_word

    return indexed_examples, metadata

  def prepro_eval(self, examples, num_chars_per_word=16):
    """Tokenizes and indexes examples (usually non-train examples).

    In order to use this, `fit` must have been already executed on train data.
    Other than that, this function has same functionality as `fit`, returning
    indexed examples.

    Args:
      examples: a list of dict, where each dict is an example.
      num_chars_per_word: `int` value, number of chars to store per word.
        This is fixed, so if word is shorter, then the rest is padded with 0.
        The charaters are flattened, so need to be reshaped when using them.
    Returns:
      a tuple `(indexed_examples, metadata)`, where `indexed_examples` is a
      list of dict (each dict being indexed example) and `metadata` is a dict
      of `glove_word2idx_dict` and statistics of the examples.
    """
    tokenized_examples = [
        _tokenize(example, self._tokenizer, num_chars_per_word)
        for example in tqdm(examples, desc='tokenizing')
    ]
    word_counter = _get_word_counter(tokenized_examples)
    glove_word2idx_dict = _get_glove_vocab(
        self._glove_words, counter=word_counter)
    tf.logging.info('Glove word vocab size: %d' % len(glove_word2idx_dict))

    def glove_word2idx(word):
      word = word.lower()
      return glove_word2idx_dict[word] if word in glove_word2idx_dict else 1

    indexed_examples = [
        _index(example, self.word2idx, glove_word2idx, self.char2idx)
        for example in tqdm(tokenized_examples, desc='indexing')
    ]

    metadata = self._get_metadata(indexed_examples)
    metadata['glove_word2idx'] = glove_word2idx_dict
    metadata['num_chars_per_word'] = num_chars_per_word
    return indexed_examples, metadata

  @property
  def savable(self):
    return {'word2idx': self._word2idx_dict, 'char2idx': self._char2idx_dict}

  def save(self, save_path):
    with tf.gfile.GFile(save_path, 'w') as fp:
      json.dump(self.savable, fp)

  def load(self, load_path):
    with tf.gfile.GFile(load_path, 'r') as fp:
      savable = json.load(fp)
    self._word2idx_dict = savable['word2idx']
    self._char2idx_dict = savable['char2idx']

  def _get_metadata(self, examples):
    metadata = {
        'max_context_size': max(len(e['context_words']) for e in examples),
        'max_ques_size': max(len(e['question_words']) for e in examples),
        'word_vocab_size': len(self._word2idx_dict),
        'char_vocab_size': len(self._char2idx_dict),
    }
    return metadata


def split(example, para2sents_fn=None, positive_augment_factor=0):
  """Splits context in example into sentences and create multiple examples.

  Args:
    example: `dict` object, each element of `get_examples()`.
    para2sents_fn: function that maps `str` to a list of `str`, splitting
      paragraph into sentences.
    positive_augment_factor: Multiply positive examples by this factor.
      For handling class imbalance problem.
  Returns:
    a list of examples, with modified fields: `id`, `context`, `answers` and
    `answer_starts`. Will add `has_answer` bool field.
  """
  if para2sents_fn is None:
    para2sents_fn = nltk.sent_tokenize
  sents = para2sents_fn(example['context'])
  sent_start_idxs = _tokens2idxs(example['context'], sents)

  context = example['context']
  examples = []
  for i, (sent, sent_start_idx) in enumerate(zip(sents, sent_start_idxs)):
    sent_end_idx = sent_start_idx + len(sent)
    e = dict(example.items())  # Copying dict content.
    e['context'] = sent
    e['id'] = '%s %d' % (e['id'], i)
    e['answers'] = []
    e['answer_starts'] = []
    for answer, answer_start in zip(example['answers'],
                                    example['answer_starts']):
      answer_end = answer_start + len(answer)
      if (sent_start_idx <= answer_start < sent_end_idx or
          sent_start_idx < answer_end <= sent_end_idx):
        new_answer = context[max(sent_start_idx, answer_start):min(
            sent_end_idx, answer_end)]
        new_answer_start = max(answer_start, sent_start_idx) - sent_start_idx
        e['answers'].append(new_answer)
        e['answer_starts'].append(new_answer_start)
    if not e['answers']:
      e['answers'].append(NO_ANSWER)
      e['answer_starts'].append(-1)
    e['num_answers'] = len(e['answers'])
    # If the list is empty, then the example has no answer.
    examples.append(e)
    if positive_augment_factor and e['answers'][0] != NO_ANSWER:
      for _ in range(positive_augment_factor):
        examples.append(e)
  return examples


def get_idx2vec_mat(glove_dir, size, glove_word2idx_dict):
  """Gets embedding matrix for given GloVe vocab."""
  glove = _get_glove(glove_dir, size=size)
  glove[PAD] = glove[UNK] = [0.0] * size
  idx2vec_dict = {idx: glove[word] for word, idx in glove_word2idx_dict.items()}
  idx2vec_mat = np.array(
      [idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
  return idx2vec_mat


def get_examples(squad_path):
  """Obtain a list of examples from official SQuAD file.

  Args:
    squad_path: path to the official SQuAD file (e.g. `train-v1.1.json`).
  Returns:
    a list of dict, where each dict is example.
  """
  with tf.gfile.GFile(squad_path, 'r') as fp:
    squad = json.load(fp)

  examples = []
  version = squad['version']
  for article in squad['data']:
    title = article['title']
    for paragraph in article['paragraphs']:
      context = paragraph['context']
      for qa in paragraph['qas']:
        question = qa['question']
        id_ = qa['id']

        answer_starts = [answer['answer_start'] for answer in qa['answers']]
        answers = [answer['text'] for answer in qa['answers']]

        example = {
            'version': version,
            'title': title,
            'context': context,
            'question': question,
            'id': id_,
            'answer_starts': answer_starts,
            'answers': answers,
            'num_answers': len(answers),
            'is_supervised': True,
        }
        example = normalize_example(example)
        examples.append(example)
  return examples


def normalize_example(example):
  n_example = dict(example.items())
  n_example['context'] = _replace_quotations(n_example['context'])
  n_example['answers'] = [_replace_quotations(a) for a in n_example['answers']]
  return n_example


def _replace_quotations(text):
  return text.replace('``', '" ').replace("''", '" ')


def _word_tokenize(text):
  # TODO(seominjoon): Consider using Stanford Tokenizer or othe tokenizers.
  return [
      word.replace('``', '"').replace("''", '"')
      for word in nltk.word_tokenize(text)
  ]


def _tokens2idxs(text, tokens):
  idxs = []
  idx = 0
  for token in tokens:
    idx = text.find(token, idx)
    assert idx >= 0, (text, tokens)
    idxs.append(idx)
    idx += len(token)
  return idxs


def _tokenize(example, text2words_fn, num_chars_per_word):
  """Tokenize each example using provided tokenizer (`text2words_fn`).

  Args:
    example: `dict` value, an example.
    text2words_fn: tokenizer.
    num_chars_per_word: `int` value, number of chars to store per word.
      This is fixed, so if word is shorter, then the rest is padded with 0.
      The charaters are flattened, so need to be reshaped when using them.
  Returns:
    `dict`, representing tokenized example.
  """
  new_example = dict(example.items())
  new_example['question_words'] = text2words_fn(example['question'])
  new_example['question_num_words'] = len(new_example['question_words'])
  new_example['context_words'] = text2words_fn(example['context'])
  new_example['context_num_words'] = len(new_example['context_words'])

  def word2chars(word):
    chars = list(word)
    if len(chars) > num_chars_per_word:
      return chars[:num_chars_per_word]
    else:
      return chars + [PAD] * (num_chars_per_word - len(chars))

  new_example['question_chars'] = list(
      itertools.chain(
          * [word2chars(word) for word in new_example['question_words']]))
  new_example['context_chars'] = list(
      itertools.chain(
          * [word2chars(word) for word in new_example['context_words']]))
  return new_example


def _index(example, word2idx_fn, glove_word2idx_fn, char2idx_fn):
  """Indexes each tokenized example, using provided vocabs.

  Args:
    example: `dict` representing tokenized example.
    word2idx_fn: indexer for word vocab.
    glove_word2idx_fn: indexer for glove word vocab.
    char2idx_fn: indexer for character vocab.
  Returns:
    `dict` representing indexed example.
  """
  new_example = dict(example.items())
  new_example['indexed_question_words'] = [
      word2idx_fn(word) for word in example['question_words']
  ]
  new_example['indexed_context_words'] = [
      word2idx_fn(word) for word in example['context_words']
  ]
  new_example['indexed_question_chars'] = [
      char2idx_fn(word) for word in example['question_chars']
  ]
  new_example['indexed_context_chars'] = [
      char2idx_fn(word) for word in example['context_chars']
  ]
  new_example['glove_indexed_question_words'] = [
      glove_word2idx_fn(word) for word in example['question_words']
  ]
  new_example['glove_indexed_context_words'] = [
      glove_word2idx_fn(word) for word in example['context_words']
  ]

  word_answer_starts = []
  word_answer_ends = []
  for answer_start, answer in zip(new_example['answer_starts'],
                                  new_example['answers']):
    if answer_start < 0:
      word_answer_starts.append(-1)
      word_answer_ends.append(-1)
      break
    word_answer_start, word_answer_end = _get_word_answer(
        new_example['context'], new_example['context_words'], answer_start,
        answer)
    word_answer_starts.append(word_answer_start)
    word_answer_ends.append(word_answer_end)
  new_example['word_answer_starts'] = word_answer_starts
  new_example['word_answer_ends'] = word_answer_ends

  return new_example


def _get_glove(glove_path, size=None, draft=False):
  """Get an `OrderedDict` that maps word to vector.

  Args:
    glove_path: `str` value,
      path to the glove file (e.g. `glove.6B.50d.txt`) or directory.
    size: `int` value, size of the vector, if `glove_path` is a directory.
    draft: `bool` value, whether to only load first 99 for draft mode.
  Returns:
    `OrderedDict` object, mapping word to vector.
  """
  if size is not None:
    glove_path = os.path.join(glove_path, 'glove.6B.%dd.txt' % size)
  glove = OrderedDict()
  with tf.gfile.GFile(glove_path, 'rb') as fp:
    for idx, line in enumerate(fp):
      line = line.decode('utf-8')
      tokens = line.strip().split(u' ')
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      glove[word] = vec
      if draft and idx > 99:
        break
  return glove


def _get_word_counter(examples):
  # TODO(seominjoon): Consider not ignoring uppercase.
  counter = Counter()
  for example in tqdm(examples, desc='word counter'):
    for word in example['question_words']:
      counter[word.lower()] += 1
    for word in example['context_words']:
      counter[word.lower()] += 1
  return counter


def _get_char_counter(examples):
  counter = Counter()
  for example in tqdm(examples, desc='char counter'):
    for chars in example['question_chars']:
      for char in chars:
        counter[char] += 1
    for chars in example['context_chars']:
      for char in chars:
        counter[char] += 1
  return counter


def _counter2vocab(counter, min_count):
  tokens = [token for token, count in counter.items() if count >= min_count]
  tokens = [PAD, UNK] + tokens
  vocab = {token: idx for idx, token in enumerate(tokens)}
  return vocab


def _get_glove_vocab(words, counter=None):
  if counter is not None:
    words = [word for word in counter if word in set(words)]
  words = [PAD, UNK] + words
  vocab = {word: idx for idx, word in enumerate(words)}
  return vocab


def _get_word_answer(context, context_words, answer_start, answer):
  """Get word-level answer index.

  Args:
    context: `unicode`, representing the context of the question.
    context_words: a list of `unicode`, tokenized context.
    answer_start: `int`, the char-level start index of the answer.
    answer: `unicode`, the answer that is substring of context.
  Returns:
    a tuple of `(word_answer_start, word_answer_end)`, representing the start
    and end indices of the answer in respect to `context_words`.
  """
  assert answer, 'Encountered length-0 answer.'
  answer_end = answer_start + len(answer)
  char_idxs = _tokens2idxs(context, context_words)
  word_answer_start = None
  word_answer_end = None
  for word_idx, char_idx in enumerate(char_idxs):
    if char_idx <= answer_start:
      word_answer_start = word_idx
    if char_idx < answer_end:
      word_answer_end = word_idx
  return word_answer_start, word_answer_end

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
"""Experiments for kernel vs feature map in SQuAD.

`feature` model does not allow any interaction between question and context
except at the end, where the dot product (or L1/L2 distance) is used to get the
answer.
`kernel` model allows any interaction between question and context
(e.g. cross attention).
This script is for establishing baseline for both feature and kernel models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json
import os

import tensorflow as tf
from tqdm import tqdm
import tensorflow.contrib.learn as learn

# This is required for importing google specific flags:
# `output_dir`, `schedule`
# (`learn` above is not sufficient). Will need to add these flags when
# removing this import for open-sourcing.
from tensorflow.contrib.learn import learn_runner

import squad_data
from common_model import get_loss
from common_model import get_pred_ops
from common_model import get_train_op
from feature_model import feature_model
from kernel_model import kernel_model

tf.flags.DEFINE_integer('emb_size', 200, 'embedding size')
tf.flags.DEFINE_integer('glove_size', 200, 'GloVe size')
tf.flags.DEFINE_integer('hidden_size', 200, 'hidden state size')
tf.flags.DEFINE_integer('num_train_steps', 20000, 'num train steps')
tf.flags.DEFINE_integer('num_eval_steps', 500, 'num eval steps')
tf.flags.DEFINE_boolean('draft', False, 'draft?')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_float('dropout_rate', 0.2,
                      'dropout rate, applied to the input of LSTMs.')
tf.flags.DEFINE_string(
    'root_data_dir',
    '/cns/ok-d/home/neon-core/ker2vec/squad/prepro/sort_filter',
    'root data dir')
tf.flags.DEFINE_integer('save_checkpoints_steps', 500, '')
tf.flags.DEFINE_integer('num_eval_delay_secs', 1, 'eval delay secs')
tf.flags.DEFINE_boolean('shuffle_examples', False, 'Use shuffle example queue?')
tf.flags.DEFINE_boolean('shuffle_files', True, 'Use shuffle file queue?')
tf.flags.DEFINE_string('model', 'feature', '`feature` or `kernel`.')
tf.flags.DEFINE_boolean('oom_test', False, 'Performs out-of-memory test')
tf.flags.DEFINE_string(
    'dist', 'dot', 'Distance function for feature model. `dot`, `l1` or `l2`.')
tf.flags.DEFINE_float('learning_rate', 0.001,
                      '(Initial) learning rate for optimizer')
tf.flags.DEFINE_boolean(
    'infer', False,
    'If `True`, obtains and saves predictions for the test dataset '
    'at `answers_path`.')
tf.flags.DEFINE_string('answers_path', '',
                       'The path for saving predictions on test dataset. '
                       'If not specified, saves in `restore_dir` directory.')
tf.flags.DEFINE_float('clip_norm', 0, 'Clip norm threshold, 0 for no clip.')
tf.flags.DEFINE_integer(
    'restore_step', 0,
    'The global step for which the model is restored in the beginning. '
    '`0` for the most recent save file.')
tf.flags.DEFINE_float(
    'restore_decay', 1.0,
    'The decay rate for exponential moving average of variables that '
    'will be restored upon eval or infer. '
    '`1.0` for restoring variables without decay.')
tf.flags.DEFINE_string(
    'ema_decays', '',
    'List of exponential moving average (EMA) decay rates (float) '
    'to track for variables during training. Values are separated by commas.')
tf.flags.DEFINE_string(
    'restore_dir', '',
    'Directory from which variables are restored. If not specfied, `output_dir`'
    'will be used instead. For inference mode, this needs to be specified.')
tf.flags.DEFINE_string('model_id', 'm00', 'Model id.')
tf.flags.DEFINE_string('glove_dir', '/cns/ok-d/home/neon-core/ker2vec/glove',
                       'GloVe dir.')
tf.flags.DEFINE_boolean('merge', False, 'If `True`, merges answers from same '
                        'paragraph that were split in preprocessing step.')
tf.flags.DEFINE_integer('queue_capacity', 5000, 'Input queue capacity.')
tf.flags.DEFINE_integer('min_after_dequeue', 1000, 'Minimum number of examples '
                        'after queue dequeue.')
tf.flags.DEFINE_integer('max_answer_size', 7, 'Max number of answer words.')
tf.flags.DEFINE_string('restore_scopes', '', 'Restore scopes, separated by ,.')
tf.flags.DEFINE_boolean('reg_gen', True, 'Whether to regularize training '
                        'with question generation (reconstruction) loss.')
tf.flags.DEFINE_float('reg_cf', 3.0, 'Regularization initial coefficient.')
tf.flags.DEFINE_float('reg_half_life', 6000, 'Regularization decay half life. '
                      'Set it to very high value to effectively disable decay.')
tf.flags.DEFINE_integer('max_gen_length', 32, 'During inference, maximum '
                        'length of generated question.')

# Below are added for third party.
tf.flags.DEFINE_string('schedule', 'train_and_evaluate',
                       'schedule for learn_runner.')
tf.flags.DEFINE_string('output_dir', '/tmp/squad_ckpts',
                       'Output directory for saving model.')

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, targets, mode, params):
  """Model function to be used for `Experiment` object.

  Should not access `flags.FLAGS`.

  Args:
    features: a dictionary of feature tensors.
    targets: a dictionary of target tensors.
    mode: `learn.ModeKeys.TRAIN` or `learn.ModeKeys.EVAL`.
    params: `HParams` object.
  Returns:
    `ModelFnOps` object.
  Raises:
    ValueError: rasied if `params.model` is not an appropriate value.
  """
  with tf.variable_scope('model'):
    if params.model == 'feature':
      logits_start, logits_end, tensors = feature_model(
          features, mode, params)
    elif params.model == 'kernel':
      logits_start, logits_end, tensors = kernel_model(
          features, mode, params)
    else:
      raise ValueError(
          '`%s` is an invalid argument for `model` parameter.' % params.model)
    no_answer_bias = tf.get_variable('no_answer_bias', shape=[], dtype='float')
    no_answer_bias = tf.tile(
        tf.reshape(no_answer_bias, [1, 1]),
        [tf.shape(features['context_words'])[0], 1])

    predictions = get_pred_ops(features, params, logits_start, logits_end,
                               no_answer_bias)
    predictions.update(tensors)
    predictions.update(features)

  if mode == learn.ModeKeys.INFER:
    eval_metric_ops, loss = None, None
  else:
    eval_metric_ops = squad_data.get_eval_metric_ops(targets, predictions)
    loss = get_loss(targets['word_answer_starts'], targets['word_answer_ends'],
                    logits_start, logits_end, no_answer_bias)

  emas = {
      decay: tf.train.ExponentialMovingAverage(
          decay=decay, name='EMA_%f' % decay)
      for decay in params.ema_decays
  }

  ema_ops = [ema.apply() for ema in emas.values()]
  if mode == learn.ModeKeys.TRAIN:
    train_op = get_train_op(
        loss,
        learning_rate=params.learning_rate,
        clip_norm=params.clip_norm,
        post_ops=ema_ops)
    # TODO(seominjoon): Checking `Exists` is not the best way to do this.
    if params.restore_dir and not tf.gfile.Exists(params.output_dir):
      assert params.restore_scopes
      checkpoint_dir = params.restore_dir
      if params.restore_step:
        checkpoint_dir = os.path.join(params.restore_dir,
                                      'model.ckpt-%d' % params.restore_step)
      restore_vars = []
      for restore_scope in params.restore_scopes:
        restore_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, restore_scope))
      assignment_map = {var.op.name: var for var in restore_vars}
      tf.contrib.framework.init_from_checkpoint(checkpoint_dir, assignment_map)
  else:
    if params.restore_decay < 1.0:
      ema = emas[params.restore_decay]
      assign_ops = []
      for var in tf.trainable_variables():
        assign_op = tf.assign(var, ema.average(var))
        assign_ops.append(assign_op)
      with tf.control_dependencies(assign_ops):
        for key, val in predictions.items():
          predictions[key] = tf.identity(val)
    train_op = None

  return learn.ModelFnOps(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def _experiment_fn(run_config, hparams):
  """Outputs `Experiment` object given `output_dir`.

  Args:
    run_config: `EstimatorConfig` object fo run configuration.
    hparams: `HParams` object that contains hyperparameters.

  Returns:
    `Experiment` object
  """
  estimator = learn.Estimator(
      model_fn=model_fn, config=run_config, params=hparams)

  num_train_steps = 1 if FLAGS.oom_test else FLAGS.num_train_steps
  num_eval_steps = 1 if FLAGS.oom_test else FLAGS.num_eval_steps

  return learn.Experiment(
      estimator=estimator,
      train_input_fn=_get_train_input_fn(),
      eval_input_fn=_get_eval_input_fn(),
      train_steps=num_train_steps,
      eval_steps=num_eval_steps,
      eval_delay_secs=FLAGS.num_eval_delay_secs)


def _get_train_input_fn():
  """Get train input function."""
  train_input_fn = squad_data.get_input_fn(
      FLAGS.root_data_dir,
      FLAGS.glove_dir,
      'train',
      FLAGS.batch_size,
      FLAGS.glove_size,
      shuffle_files=FLAGS.shuffle_files,
      shuffle_examples=FLAGS.shuffle_examples,
      queue_capacity=FLAGS.queue_capacity,
      min_after_dequeue=FLAGS.min_after_dequeue,
      oom_test=FLAGS.oom_test)
  return train_input_fn


def _get_eval_input_fn():
  """Get eval input function."""
  eval_input_fn = squad_data.get_input_fn(
      FLAGS.root_data_dir,
      FLAGS.glove_dir,
      'dev',
      FLAGS.batch_size,
      FLAGS.glove_size,
      shuffle_files=True,
      shuffle_examples=True,
      queue_capacity=FLAGS.queue_capacity,
      min_after_dequeue=FLAGS.min_after_dequeue,
      num_epochs=1,
      oom_test=FLAGS.oom_test)
  return eval_input_fn


def _get_test_input_fn():
  """Get test input function."""
  # TODO(seominjoon) For now, test input is same as eval input (dev).
  test_input_fn = squad_data.get_input_fn(
      FLAGS.root_data_dir,
      FLAGS.glove_dir,
      'dev',
      FLAGS.batch_size,
      FLAGS.glove_size,
      shuffle_files=FLAGS.shuffle_files,
      shuffle_examples=FLAGS.shuffle_examples,
      queue_capacity=FLAGS.queue_capacity,
      min_after_dequeue=FLAGS.min_after_dequeue,
      num_epochs=1,
      oom_test=FLAGS.oom_test)
  return test_input_fn


def _get_config():
  """Get configuration object  for `Estimator` object.

  For open-soucing, `EstimatorConfig` has been replaced with `RunConfig`.
  Depends on `flags.FLAGS`, and should not be used outside of this main script.

  Returns:
    `EstimatorConfig` object.
  """
  config = learn.RunConfig(
      model_dir=FLAGS.restore_dir if FLAGS.infer else FLAGS.output_dir,
      keep_checkpoint_max=0,  # Keep all checkpoints.
      save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  return config


def _get_hparams():
  """Model-specific hyperparameters go here.

  All model parameters go here, since `model_fn()` should not access
  `flags.FLAGS`.
  Depends on `flags.FLAGS`, and should not be used outside of this main script.

  Returns:
    `HParams` object.
  """
  hparams = tf.contrib.training.HParams()
  data_hparams = squad_data.get_params(FLAGS.root_data_dir)
  hparams.vocab_size = data_hparams['vocab_size']
  hparams.char_vocab_size = data_hparams['char_vocab_size']
  hparams.batch_size = FLAGS.batch_size
  hparams.hidden_size = FLAGS.hidden_size
  hparams.emb_size = FLAGS.emb_size
  hparams.dropout_rate = FLAGS.dropout_rate
  hparams.dist = FLAGS.dist
  hparams.learning_rate = FLAGS.learning_rate
  hparams.model = FLAGS.model
  hparams.restore_dir = FLAGS.restore_dir
  hparams.output_dir = FLAGS.output_dir
  hparams.clip_norm = FLAGS.clip_norm
  hparams.restore_decay = FLAGS.restore_decay
  if FLAGS.ema_decays:
    hparams.ema_decays = list(map(float, FLAGS.ema_decays.split(',')))
  else:
    hparams.ema_decays = []
  hparams.restore_step = FLAGS.restore_step
  hparams.model_id = FLAGS.model_id
  hparams.max_answer_size = FLAGS.max_answer_size
  hparams.restore_scopes = FLAGS.restore_scopes.split(',')
  hparams.glove_size = FLAGS.glove_size

  # Regularization by Query Generation (reconstruction) parameters.
  hparams.reg_gen = FLAGS.reg_gen
  hparams.reg_cf = FLAGS.reg_cf
  hparams.reg_half_life = FLAGS.reg_half_life

  return hparams


def train_and_eval():
  """Train and eval routine."""
  learn_runner.run(
      experiment_fn=_experiment_fn,
      schedule=FLAGS.schedule,
      run_config=_get_config(),
      hparams=_get_hparams())


def _set_ckpt():
  # TODO(seominjoon): This is adhoc. Need better ckpt loading during inf.
  if FLAGS.restore_step:
    path = os.path.join(FLAGS.restore_dir, 'checkpoint')
    with tf.gfile.GFile(path, 'w') as fp:
      fp.write('model_checkpoint_path: "model.ckpt-%d"\n' % FLAGS.restore_step)


def infer():
  """Inference routine, outputting answers to `FLAGS.answers_path`."""
  _set_ckpt()
  estimator = learn.Estimator(
      model_fn=model_fn, config=_get_config(), params=_get_hparams())
  predictions = estimator.predict(
      input_fn=_get_test_input_fn(), as_iterable=True)
  global_step = estimator.get_variable_value('global_step')
  path = FLAGS.answers_path or os.path.join(FLAGS.restore_dir,
                                            'answers-%d.json' % global_step)
  answer_dict = {'no_answer_prob': {}, 'answer_prob': {}}
  for prediction in tqdm(predictions):
    id_ = prediction['id'].decode('utf-8')
    answer_dict[id_] = prediction['a'].decode('utf-8')
    answer_dict['answer_prob'][id_] = prediction['answer_prob'].tolist()
    answer_dict['no_answer_prob'][id_] = prediction['no_answer_prob'].tolist()
    if FLAGS.oom_test:
      break

  # TODO(seominjoon): use sum of logits instead of normalized prob.
  if FLAGS.merge:
    new_answer_dict = defaultdict(list)
    for id_, answer_prob in answer_dict['answer_prob'].items():
      answer = answer_dict[id_]
      id_ = id_.split(' ')[0]  # retrieve true id
      new_answer_dict[id_].append([answer_prob, answer])
    answer_dict = {
        id_: max(each, key=lambda pair: pair[0])[1]
        for id_, each in new_answer_dict.items()
    }

  with tf.gfile.GFile(path, 'w') as fp:
    json.dump(answer_dict, fp)
  tf.logging.info('Dumped predictions at: %s' % path)


def main(_):
  if FLAGS.infer:
    infer()
  else:
    train_and_eval()


if __name__ == '__main__':
  tf.app.run()

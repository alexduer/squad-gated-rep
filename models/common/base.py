import copy
import json
import os
from typing import Any, Dict, List, Tuple

import tensorflow as tf
import numpy as np

from common.data.dataset.proc_dataset import ProcDataset
from common.data.evaluate import evaluate
from common.data.pipeline.embedder import load_word_embedder, load_small_char_embedder, load_char_embedder
from common.data.pipeline.iterator import BatchIter, SampleBatch
from common.util.cache import cache
from common.util.printing import print_epoch_summary, ordinal
from common.util.time import now


class BaseModel(object):
    def __init__(self, config: Dict[str, Any], datasets: List[ProcDataset]):
        self.config = copy.deepcopy(config)
        print('running with config: ', self.config)

        self.conf_random_seed = int(self.config['random_seed'])
        self.conf_filter_short_questions = bool(self.config['filter_short_questions'])
        self.conf_save_meta_graph = bool(self.config['save_meta_graph'])
        self.conf_patience = int(self.config['patience'])

        self.conf_layer_size = int(self.config['layer_size'])
        self.conf_att_size = int(self.config['att_size'])
        self.conf_char_rnn_size = int(self.config['char_rnn_size'])
        self.conf_dropout = float(self.config['dropout'])

        self.conf_max_par_length = int(self.config['max_par_length'])
        self.conf_max_qu_length = int(self.config['max_qu_length'])
        self.conf_max_char_length = int(self.config['max_char_length'])
        self.conf_rnn_parallelity = int(self.config['rnn_parallelity'])
        self.conf_iter_bucket_size = int(self.config['iter_bucket_size'])

        self.conf_batch_size = int(self.config['batch_size'])
        self.conf_apply_grads_interval = int(self.config['apply_grads_interval'])

        self.conf_optimizer = self.config['optimizer']
        self.conf_opt_lr = float(self.config['opt_lr'])
        self.conf_opt_epsilon = float(self.config['opt_epsilon'])
        self.conf_opt_adam_beta1 = float(self.config['opt_adam_beta1'])
        self.conf_opt_adam_beta2 = float(self.config['opt_adam_beta2'])
        self.conf_opt_adadelta_rho = float(self.config['opt_adadelta_rho'])
        self.conf_lr_reduction_criterion = self.config['lr_reduction_criterion'].upper()
        if self.conf_lr_reduction_criterion.upper() != 'F1' and self.conf_lr_reduction_criterion != 'EM':
            raise Exception('illegal value for parameter lr_reduction_criterion: {}, allowed values: F1, EM'
                            .format(self.conf_lr_reduction_criterion))

        self.conf_cache_dir = self.config['cache_dir']
        self.conf_weights_dir = self.config['weights_dir']
        self.conf_placement = self.config['placement']

        self.word_embedder = cache(lambda: load_word_embedder(datasets, self.config['data_dir']),
                                   self.config['cache_dir'], '{}_{}_{}_{}_{}'.format(
                len(datasets), len(datasets[0].documents), len(datasets[-1].documents),
                datasets[0].documents[0].raw.title, datasets[-1].documents[-1].raw.title
            ))

        if bool(self.config['load_embeddings_from_file']):
            self.char_embedder = cache(lambda: load_char_embedder(self.config['data_dir']),
                                       self.config['cache_dir'], 'char_embedder')
        else:
            self.char_embedder = cache(lambda: load_small_char_embedder(self.config['data_dir'], self.conf_layer_size),
                                       self.config['cache_dir'], 'small_char_embedder')

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_resets = tf.Variable(initial_value=0, name='num_resets', trainable=False, dtype=tf.int32)

        # child functions
        self.dropout_keep_prob = None
        self.predictions = None
        self.loss = None
        self.graph = self.build_graph(self.graph)
        self.check_interface()

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.session = tf.Session(config=sess_config, graph=self.graph)

        with self.graph.as_default():
            self.bad_iter_count = tf.Variable(initial_value=0, name='bad_iter_count', trainable=False, dtype=tf.int32)
            self.increment_bad_iter_count_op = tf.assign(self.bad_iter_count,
                                                         tf.add(self.bad_iter_count, tf.constant(1, dtype=tf.int32)))

            self.best_f1_score = tf.Variable(initial_value=-1.0, name='best_f1_score', trainable=False, dtype=tf.float32)
            self.best_em_score = tf.Variable(initial_value=-1.0, name='best_em_score', trainable=False, dtype=tf.float32)
            self.prev_f1_score = tf.Variable(initial_value=-1.0, name='prev_f1_score', trainable=False, dtype=tf.float32)
            self.prev_em_score = tf.Variable(initial_value=-1.0, name='prev_em_score', trainable=False, dtype=tf.float32)
            self.best_epoch_id = tf.Variable(initial_value=1, name='best_epoch_id', trainable=False, dtype=tf.int32)

            static_graph_vars, saveable_graph_vars, optimizer_vars = self.var_groups()
            self.saver = tf.train.Saver(var_list=[*saveable_graph_vars, *optimizer_vars], save_relative_paths=True)
            self.static_var_init_op = tf.variables_initializer(static_graph_vars)
            self.saveable_var_init_op = tf.variables_initializer(saveable_graph_vars)
            self.optimizer_init_op = tf.variables_initializer(optimizer_vars)

        self.print_trainable_params()
        self.loaded_epoch_id = self.init_weights()

    def check_interface(self):
        if self.dropout_keep_prob is None:
            raise Exception('self.dropout_keep_prob is None, needs to be set in child model')
        if self.predictions is None:
            raise Exception('self.predictions is None, needs to be set in child model')
        if self.loss is None:
            raise Exception('self.loss is None, needs to be set in child model')

    def build_graph(self, graph) -> tf.Graph:
        raise NotImplementedError('The build_graph function must be overriden by the specialized model')

    def build_inputs(self):
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')

        self.par_words = tf.placeholder(name='input_par_words', dtype=tf.int32,
                                        shape=[self.conf_batch_size, None])
        self.par_num_words = tf.placeholder(name='input_par_num_words', dtype=tf.int32,
                                            shape=[self.conf_batch_size])
        self.par_chars = tf.placeholder(name='input_par_chars', dtype=tf.int32,
                                        shape=[self.conf_batch_size, None, None])
        self.par_num_chars = tf.placeholder(name='input_par_num_chars', dtype=tf.int32,
                                            shape=[self.conf_batch_size, None])

        self.qu_words = tf.placeholder(name='input_qu_words', dtype=tf.int32,
                                       shape=[self.conf_batch_size, None])
        self.qu_num_words = tf.placeholder(name='input_qu_num_words', dtype=tf.int32,
                                           shape=[self.conf_batch_size])
        self.qu_chars = tf.placeholder(name='input_qu_chars', dtype=tf.int32,
                                       shape=[self.conf_batch_size, None, None])
        self.qu_num_chars = tf.placeholder(name='input_qu_num_chars', dtype=tf.int32,
                                           shape=[self.conf_batch_size, None])

        self.answer_labels = tf.placeholder(name='input_answer_labels', dtype=tf.int32,
                                            shape=[self.conf_batch_size, 2])

    def build_optimizer(self, learning_rate: float, reuse: bool=False) -> None:
        print('building optimizer {} with learning rate {} and reuse {}'
              .format(self.conf_optimizer, learning_rate, reuse))

        with tf.variable_scope('OPTIMIZER', reuse=reuse):
            if self.conf_optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                        beta1=self.conf_opt_adam_beta1, beta2=self.conf_opt_adam_beta2,
                                                        epsilon=self.conf_opt_epsilon)
            elif self.conf_optimizer == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                            rho=self.conf_opt_adadelta_rho,
                                                            epsilon=self.conf_opt_epsilon)

            self.optimizer_op = self.optimizer.minimize(self.loss)

    def reset_optimizer(self, learning_rate: float, hard_reset: bool) -> None:
        print('resetting optimizer, with learning rate {}, hard reset: {}'.format(learning_rate, hard_reset))
        with self.graph.as_default():
            self.build_optimizer(learning_rate, True)
            _, _, optimizer_vars = self.var_groups()
            self.optimizer_init_op = tf.variables_initializer(optimizer_vars)

            # relevant only for optimizers like Adam or Adadelta that keep track of moments per variable
            if hard_reset:
                self.session.run([self.optimizer_init_op])

    def var_groups(self) -> Tuple[List[tf.Variable], List[tf.Variable], List[tf.Variable]]:
        static_graph_vars = []
        saveable_graph_vars = []
        optimizer_vars = []

        for var in tf.global_variables():
            if 'OPTIMIZER' in var.name:
                optimizer_vars.append(var)
            elif 'word_embeddings' in var.name:
                static_graph_vars.append(var)
            else:
                saveable_graph_vars.append(var)

        return static_graph_vars, saveable_graph_vars, optimizer_vars

    def print_trainable_params(self) -> None:
        with self.graph.as_default():
            total_params = 0
            for var in tf.trainable_variables():
                shape = var.get_shape()

                var_params = 1
                for dim in shape:
                    var_params *= dim.value

                total_params += var_params

            print('[{}] there are {} total trainable parameters in this model'.format(now(), total_params))

    def to_feed_dict(self, batch: SampleBatch, use_dropout: bool) -> Tuple[List[str], Dict[tf.Variable, np.array]]:
        if use_dropout:
            dropout_keep_prob = self.conf_dropout
        else:
            dropout_keep_prob = 1.0

        return batch.qu_ids, {
            self.dropout_keep_prob: dropout_keep_prob,
            self.par_words: batch.par_words,
            self.par_num_words: batch.par_num_words,
            self.par_chars: batch.par_chars,
            self.par_num_chars: batch.par_num_chars,
            self.qu_words: batch.qu_words,
            self.qu_num_words: batch.qu_num_words,
            self.qu_chars: batch.qu_chars,
            self.qu_num_chars: batch.qu_num_chars,
            self.answer_labels: batch.answer_labels,
        }

    def init_weights(self) -> int:
        checkpoint = tf.train.get_checkpoint_state(self.conf_weights_dir)
        with self.graph.as_default():
            self.session.run(self.static_var_init_op)

        if checkpoint is not None:
            print('Loading the model {} from folder {}'.format(checkpoint.model_checkpoint_path, self.conf_weights_dir))
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            self.reset_optimizer(self.conf_opt_lr / 2.0 ** (self.session.run(self.num_resets)), False)
            return int(checkpoint.model_checkpoint_path.split('-')[-1].split('.')[0])
        else:
            print('No existing checkpoint found in folder: %s' % self.conf_weights_dir)
            self.session.run([self.saveable_var_init_op, self.optimizer_init_op])
            return -1

    def load_weights(self, epoch_id: int) -> None:
        save_path = os.path.join(self.conf_weights_dir, 'model-{}'.format(epoch_id))
        print('Loading the model from file: {}'.format(save_path))
        self.session.run(self.static_var_init_op)
        self.saver.restore(self.session, save_path)

    def train_epoch(self, dataset: ProcDataset, epoch_id: int) -> None:
        batch_iter = self.create_iter(dataset, True)

        num_batches = batch_iter.num_valid_batches()
        info_interval = num_batches // 10

        batch_counter, loss_sum = 0, 0.0

        for batch_id in range(num_batches):
            _, feed_dict = self.to_feed_dict(batch_iter.__next__(), True)

            iterations = (batch_id + 1) * self.conf_batch_size
            loss_val, _ = self.session.run([self.loss, self.optimizer_op], feed_dict=feed_dict)

            batch_counter += 1
            # noinspection PyTypeChecker
            loss_sum += float(np.sum(loss_val, 0))
            if (batch_id + 1) % info_interval == 0:
                train_loss = loss_sum / batch_counter
                print('[{} | {} | {}] train loss: {}'.format(now(), epoch_id, iterations, train_loss))
                batch_counter, loss_sum = 0, 0.0

    def train(self, train_dataset: ProcDataset, valid_dataset: ProcDataset, max_epochs=1000, patience=0) -> None:
        for epoch_id in range(self.loaded_epoch_id + 1, max_epochs):
            print("[{} | {}] current best em: {}, current best f1: {}, current bad iter count {}, current num resets {}"
                  .format(now(), epoch_id, *self.session.run([self.best_em_score, self.best_f1_score,
                                                              self.bad_iter_count, self.num_resets])))
            self.train_epoch(train_dataset, epoch_id)

            em_score, f1_score = evaluate(valid_dataset, self.infer(valid_dataset))
            print_epoch_summary(epoch_id, em_score, f1_score)

            self.session.run(self.increment_bad_iter_count_op)
            bad_iter_count, best_f1_score, best_em_score, best_epoch_id, num_resets = \
                self.session.run([self.bad_iter_count, self.best_f1_score, self.best_em_score,
                                  self.best_epoch_id, self.num_resets])

            if f1_score > best_f1_score and self.conf_lr_reduction_criterion == 'F1' or \
                                    em_score > best_em_score and self.conf_lr_reduction_criterion == 'EM':
                print('[{} | {}] new best iteration!'.format(now(), epoch_id))
                self.session.run([
                    tf.assign(self.bad_iter_count, 0),
                    tf.assign(self.best_f1_score, f1_score),
                    tf.assign(self.best_em_score, em_score),
                    tf.assign(self.best_epoch_id, epoch_id),
                ])
            elif bad_iter_count > patience:
                num_resets = self.session.run(self.num_resets)
                print('[{} | {}] resetting the {} time'.format(now(), epoch_id, ordinal(num_resets + 1)))
                self.init_weights()
                self.session.run([
                    tf.assign(self.bad_iter_count, 0),
                    tf.assign(self.num_resets, num_resets + 1),
                ])

                self.reset_optimizer(self.conf_opt_lr / 2.0 ** (num_resets + 1), True)
            else:
                print('[{} | {}] bad iteration, not doing anything yet, curr_patience {}, patience'
                      .format(now(), bad_iter_count, patience))

            self.save(epoch_id)

    def save(self, epoch_id) -> str:
        tf.logging.set_verbosity('DEBUG')
        model_file_stump = os.path.join(self.conf_weights_dir, 'model')
        print('[{}] Saving to {} with epoch_id {}'.format(now(), model_file_stump, epoch_id))
        save_path = self.saver.save(self.session, model_file_stump, epoch_id,
                                    write_meta_graph=self.conf_save_meta_graph)

        conf_json = os.path.join(self.conf_weights_dir, 'config.json')
        if not os.path.isfile(conf_json):
            with open(conf_json, 'w') as f:
                json.dump(self.config, f)

        print('[{}] finished saving!'.format(now()))
        tf.logging.set_verbosity('WARN')
        return save_path

    def infer(self, dataset: ProcDataset) -> Dict[str, str]:
        print('[{}] starting inference ...'.format(now()))
        dataset_iter = self.create_iter(dataset, False)
        num_samples = dataset_iter.num_samples()
        index_results = {}

        for first_sample_index in range(0, num_samples + self.conf_batch_size, self.conf_batch_size):
            qu_ids, feed_dict = self.to_feed_dict(dataset_iter.__next__(), False)
            pred_val = self.session.run(self.predictions, feed_dict=feed_dict)

            for i in range(0, self.conf_batch_size):
                # noinspection PyTypeChecker
                index_results[qu_ids[i]] = (int(np.argmax(pred_val[i, 0])), int(np.argmax(pred_val[i, 1])))

        text_results = {}
        for doc in dataset.documents:
            for par in doc.paragraphs:
                for qu in par.questions:
                    first_token_index, last_token_index = index_results[qu.raw.id]
                    first_token_index = min([first_token_index, len(par.tokens) - 1])
                    last_token_index = min([last_token_index, len(par.tokens) - 1])

                    char_offset_start = par.tokens[first_token_index].char_offset
                    char_offset_end = par.tokens[last_token_index].char_offset_end()
                    text_results[qu.raw.id] = par.raw.context[char_offset_start:char_offset_end]

        return text_results

    def create_iter(self, dataset: ProcDataset, discard_invalid_samples: bool) -> BatchIter:
        return BatchIter(dataset, self.word_embedder, self.char_embedder, self.conf_batch_size,
                         self.conf_max_par_length, self.conf_max_qu_length, self.conf_max_char_length,
                         discard_invalid_samples, self.conf_filter_short_questions, self.conf_iter_bucket_size)

    def apply_dropout(self, var: tf.Variable) -> tf.Variable:
        return tf.nn.dropout(var, self.dropout_keep_prob)

    def apply_dropout_to_list(self, vars: List[tf.Variable]) -> List[tf.Variable]:
        return [self.apply_dropout(v) for v in vars]

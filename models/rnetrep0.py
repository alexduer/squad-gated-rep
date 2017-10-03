from typing import Tuple

import numpy as np
import tensorflow as tf

from common.data.dataset.proc_dataset import ProcDataset
from common.data.evaluate import evaluate
from common.util.printing import print_var, print_batch_summary, ordinal
from common.util.time import now
from models.common.base import BaseModel
from models.common.components import loss_neg_log_prob
from models.common.embedding import embedding_layer
from models.common.rnet import rnet_prediction, rnet_matching_layer, rnet_self_matching_layer_real
from models.common.rnn import bi_gru_layer


class RnetRep0(BaseModel):
    # noinspection PyAttributeOutsideInit
    def build_graph(self, graph) -> tf.Graph:
        print('[{}] building {} model ...'.format(now(), self.__class__.__name__))

        with graph.as_default():
            self.build_inputs()

            # building the actual model
            with tf.device('/gpu'):
                self.par_vectors = self.apply_dropout(
                    embedding_layer(self.word_embedder, self.char_embedder, self.conf_layer_size,
                                    self.par_words, self.par_num_words, self.par_chars, self.par_num_chars,
                                    False, self.apply_dropout))

                self.qu_vectors = self.apply_dropout(
                    embedding_layer(self.word_embedder, self.char_embedder, self.conf_layer_size,
                                    self.qu_words, self.qu_num_words, self.qu_chars, self.qu_num_chars,
                                    True, self.apply_dropout))

                print_var('par_vectors', self.par_vectors)
                print_var('qu_vectors', self.qu_vectors)

                self.par_encoded, self.qu_encoded = self.encoding_layers()

                print_var('par_encoded', self.par_encoded)
                print_var('question_encoded', self.qu_encoded)

                self.match_par_qu = self.match_par_qu_layer()
                print_var('match_par_qu', self.match_par_qu)

                self.match_self = self.match_self_layer()
                print_var('match_self', self.match_self)

                self.predictions = self.prediction_layer()
                print_var('predictions', self.predictions)

                self.loss = self.loss_function()
                print_var('loss', self.loss)

                self.build_optimizer(self.conf_opt_lr)

        return graph

    def encoding_layers(self) -> Tuple[tf.Variable, tf.Variable]:
        with tf.variable_scope('par'):
            par_encoded = self.apply_dropout(self.encoding_layer(self.par_vectors, self.par_num_words, False))

        with tf.variable_scope('qu'):
            qu_encoded = self.apply_dropout(self.encoding_layer(self.qu_vectors, self.qu_num_words, False))

        return par_encoded, qu_encoded

    def encoding_layer(self, vecs: tf.Variable, num_words: tf.Variable, reuse: bool) -> tf.Variable:
        with tf.variable_scope('encoding', reuse=reuse):
            encoded_vecs, _, _ = bi_gru_layer([self.conf_layer_size] * 3, vecs, num_words, self.apply_dropout,
                                              self.conf_rnn_parallelity)
        return encoded_vecs

    def match_par_qu_layer(self) -> tf.Variable:
        return self.apply_dropout(
            rnet_matching_layer(self.conf_layer_size, self.conf_att_size, self.par_encoded,
                                self.qu_encoded, self.par_num_words, self.conf_rnn_parallelity))

    def match_self_layer(self) -> tf.Variable:
        return self.apply_dropout(
            rnet_self_matching_layer_real(self.conf_layer_size, self.conf_att_size, self.match_par_qu,
                                          self.par_num_words, self.conf_rnn_parallelity))

    def prediction_layer(self) -> tf.Variable:
        return rnet_prediction(self.match_self, self.qu_encoded, self.conf_layer_size,
                               self.conf_att_size)

    def loss_function(self) -> tf.Variable:
        return loss_neg_log_prob(self.predictions, self.answer_labels)

    def train_epoch(self, dataset: ProcDataset, epoch_id: int) -> None:
        batch_iter = self.create_iter(dataset, True)

        num_batches = batch_iter.num_valid_batches()
        info_interval = num_batches // 5

        batch_counter, loss_sum = 0, 0.0

        self.session.run(self.zero_ops)
        for batch_id in range(num_batches):
            _, feed_dict = self.to_feed_dict(batch_iter.__next__(), True)

            iterations = (batch_id + 1) * self.conf_batch_size
            curr_lr, loss_val, _ = self.session.run([self.curr_lr, self.loss, self.accum_ops], feed_dict=feed_dict)

            if (batch_id + 1) % self.conf_apply_grads_interval == 0:
                self.session.run(self.apply_grads)
                self.session.run(self.zero_ops)

            batch_counter += 1
            # noinspection PyTypeChecker
            loss_sum += float(np.sum(loss_val, 0))
            if (batch_id + 1) % info_interval == 0:
                train_loss = loss_sum / batch_counter
                print('[{} | {} | {}] train loss: {}, lr: {}'.format(now(), epoch_id, iterations, train_loss, curr_lr))
                batch_counter, loss_sum = 0, 0.0

                num_resets = self.evaluate_batch(epoch_id)
                if num_resets >= self.conf_patience:
                    return

    def evaluate_batch(self, epoch_id) -> int:
        self.session.run(self.increment_bad_iter_count_op)

        bad_iter_count, best_f1_score, best_em_score, prev_f1_score, prev_em_score, best_epoch_id, num_resets = \
            self.session.run([self.bad_iter_count, self.best_f1_score, self.best_em_score,
                              self.prev_f1_score, self.prev_em_score, self.best_epoch_id, self.num_resets])

        em_score, f1_score = evaluate(self.valid_dataset, self.infer(self.valid_dataset))
        print_batch_summary(epoch_id, em_score, f1_score)

        if f1_score > prev_f1_score and self.conf_lr_reduction_criterion == 'F1' or \
                                em_score > prev_em_score and self.conf_lr_reduction_criterion == 'EM':
            print('[{} | {}] new good iteration!'.format(now(), epoch_id))
            self.session.run([
                tf.assign(self.bad_iter_count, 0),
                tf.assign(self.best_f1_score, f1_score),
                tf.assign(self.best_em_score, em_score),
                tf.assign(self.best_epoch_id, epoch_id),
            ])
        else:
            num_resets = self.session.run(self.num_resets)
            print('[{} | {}] bad iteration, resetting the {} time'.format(now(), epoch_id, ordinal(num_resets + 1)))
            self.session.run([
                tf.assign(self.bad_iter_count, 0),
                tf.assign(self.num_resets, num_resets + 1),
            ])
            self.reset_optimizer(self.conf_opt_lr, False)

        self.session.run([
            tf.assign(self.prev_f1_score, f1_score),
            tf.assign(self.prev_em_score, em_score),
        ])
        return self.session.run(self.num_resets)

    def train(self, train_dataset: ProcDataset, valid_dataset: ProcDataset, max_epochs=1000, patience=0) -> None:
        self.valid_dataset = valid_dataset

        for epoch_id in range(self.loaded_epoch_id + 1, max_epochs):
            em, f1, bad_iter, num_resets, curr_lr = self.session.run(
                [self.best_em_score, self.best_f1_score, self.bad_iter_count, self.num_resets, self.curr_lr])
            print("[{} | {}] current best em: {}, current best f1: {}, current bad iter count {}, "
                  "current num resets {}, current lr {}"
                  .format(now(), epoch_id, em, f1, bad_iter, num_resets, curr_lr))

            if num_resets >= self.conf_patience:
                print('patience reached, exiting ...')
                return

            self.train_epoch(train_dataset, epoch_id)

            self.save(epoch_id)

    def build_optimizer(self, learning_rate: float, reuse: bool=False) -> None:
        print('building optimizer {} with learning rate {} and reuse {}'
              .format(self.conf_optimizer, learning_rate, reuse))

        with tf.variable_scope('OPTIMIZER', reuse=reuse):
            self.curr_lr = tf.multiply(learning_rate, tf.pow(0.5, tf.cast(self.num_resets, tf.float32)))
            if self.conf_optimizer == 'Adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.curr_lr,
                                                        beta1=self.conf_opt_adam_beta1, beta2=self.conf_opt_adam_beta2,
                                                        epsilon=self.conf_opt_epsilon)
            elif self.conf_optimizer == 'Adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.curr_lr,
                                                            rho=self.conf_opt_adadelta_rho,
                                                            epsilon=self.conf_opt_epsilon)

            trainable_vars = tf.trainable_variables()
            accum_vars = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in trainable_vars]

            self.zero_ops = [var.assign(tf.zeros_like(var)) for var in accum_vars]

            gradient_vars = self.optimizer.compute_gradients(self.loss, trainable_vars)
            self.accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gradient_vars)]
            self.apply_grads = self.optimizer.apply_gradients([(tf.div(accum_vars[i], self.conf_apply_grads_interval), gv[1])
                                                               for i, gv in enumerate(gradient_vars)])

    def reset_optimizer(self, learning_rate: float, hard_reset: bool) -> None:
        print('resetting optimizer, with learning rate {}, hard reset: {}'.format(learning_rate, hard_reset))
        with self.graph.as_default():
            # relevant only for optimizers like Adam or Adadelta that keep track of moments per variable
            if hard_reset:
                self.build_optimizer(learning_rate, True)
                _, _, optimizer_vars = self.var_groups()
                self.optimizer_init_op = tf.variables_initializer(optimizer_vars)
                self.session.run([self.optimizer_init_op])

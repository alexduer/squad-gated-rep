from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from models.common.match_rnn_cell import MatchRNNCell
from models.common.rnn import bi_gru_layer
from models.rnetrep0 import RnetRep0

import tensorflow as tf


class RnetRep11(RnetRep0):
    def encoding_layers(self):
        par_encoded = self.apply_dropout(self.encoding_layer(self.par_vectors, self.par_num_words, False))
        qu_encoded = self.apply_dropout(self.encoding_layer(self.qu_vectors, self.qu_num_words, True))
        return par_encoded, qu_encoded

    def encoding_layer(self, vecs: tf.Variable, num_words: tf.Variable, reuse: bool) -> tf.Variable:
        with tf.variable_scope('encoding', reuse=reuse):
            encoded_vecs, _, _ = bi_gru_layer([self.conf_layer_size], vecs, num_words, self.apply_dropout,
                                              self.conf_rnn_parallelity)
        return encoded_vecs

    def match_par_qu_layer(self):
        with tf.variable_scope('alignment_par_qu') as scope:
            rnn_cell = MatchRNNCell(GRUCell(self.conf_layer_size), self.qu_encoded, self.conf_att_size)

            outputs, final_state = dynamic_rnn(rnn_cell, self.par_encoded, self.par_num_words,
                                               parallel_iterations=self.conf_rnn_parallelity,
                                               scope=scope, swap_memory=True, dtype=tf.float32)

            with tf.variable_scope('encoding'):
                outputs, _, _ = bi_gru_layer([self.conf_layer_size]*3, self.apply_dropout(outputs), self.par_num_words,
                                             self.apply_dropout)

        return outputs

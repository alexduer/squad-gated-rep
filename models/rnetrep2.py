from models.common.rnn import bi_gru_layer
from models.rnetrep0 import RnetRep0

import tensorflow as tf


class RnetRep2(RnetRep0):
    def encoding_layer(self, vecs: tf.Variable, num_words: tf.Variable, reuse: bool) -> tf.Variable:
        with tf.variable_scope('encoding', reuse=reuse):
            encoded_vecs, _, _ = bi_gru_layer([self.conf_layer_size], vecs, num_words, self.apply_dropout,
                                              self.conf_rnn_parallelity)
        return encoded_vecs

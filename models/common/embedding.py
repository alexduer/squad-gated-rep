from typing import Callable

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from common.data.pipeline.embedder import EmbeddingService


def embedding_layer(word_embedder: EmbeddingService, char_embedder: EmbeddingService, char_rnn_size: int,
                    words: tf.Variable, num_words: tf.Variable, chars: tf.Variable, num_chars: tf.Variable,
                    reuse: bool, dropout_function: Callable[[tf.Variable], tf.Variable]) -> tf.Variable:
    with tf.variable_scope('embedding_layer', reuse=reuse):
        word_embed = _word_embedding_layer(word_embedder, words)
        char_embed = _char_embedding_layer(char_embedder, chars, num_words, num_chars, char_rnn_size, dropout_function)
        embed = tf.concat([word_embed, char_embed], 2)
    return embed


def _word_embedding_layer(embedder: EmbeddingService, words: tf.Variable) -> tf.Variable:
    with tf.variable_scope('word_embedding_layer'):
        word_embeddings = tf.get_variable(name='word_embeddings', trainable=False, dtype=tf.float32,
                                          initializer=tf.constant(embedder.embedding_matrix, dtype=tf.float32))
        word_embed = tf.nn.embedding_lookup(word_embeddings, words)

    return word_embed


def _char_embedding_layer(embedder: EmbeddingService, chars: tf.Variable, num_words: tf.Variable,
                          num_chars: tf.Variable, char_rnn_size: int,
                          dropout_function: Callable[[tf.Variable], tf.Variable]) -> tf.Variable:
    batch_size = int(chars.get_shape()[0])
    embedding_size = embedder.embedding_dim

    with tf.variable_scope('char_embedding_layer'):
        # [batch_size, dim_num_words, dim_num_chars]
        char_embeddings = tf.get_variable(name='char_embeddings', trainable=True, dtype=tf.float32,
                                          initializer=tf.constant(embedder.embedding_matrix, dtype=tf.float32))
        char_raw_embed = dropout_function(tf.nn.embedding_lookup(char_embeddings, chars))

        # we need to unstack instead of reshape as two dimension are unknown
        # batch_size * [dim_num_words, dim_num_chars, embedding_size]
        char_raw_embed_list = tf.unstack(char_raw_embed, batch_size, axis=0)
        char_raw_embed_length_list = tf.unstack(num_chars, batch_size, axis=0)
        # batch_size * [dim_num_words, layer_size]
        char_embed_list = []

        with tf.variable_scope('encoding') as scope:
            fw_cell = GRUCell(char_rnn_size)
            bw_cell = GRUCell(char_rnn_size)

            for i in range(len(char_raw_embed_list)):
                batch_embed = char_raw_embed_list[i]
                batch_char_length = char_raw_embed_length_list[i]

                (_, _), (fw_final, bw_final) = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=batch_embed,
                                                                         dtype=tf.float32,
                                                                         sequence_length=batch_char_length,
                                                                         scope=scope, parallel_iterations=64,
                                                                         swap_memory=True)
                out = tf.concat([fw_final, bw_final], axis=1)
                char_embed_list.append(out)

    return tf.stack(char_embed_list, axis=0)

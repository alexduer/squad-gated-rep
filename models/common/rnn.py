from typing import Tuple, List, Callable

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell


def bi_gru_layer(layer_sizes: List[int], input: tf.Variable, input_length: tf.Variable,
                 dropout_function: Callable[[tf.Variable], tf.Variable]=None,
                 parallel_iterations: int=64) -> Tuple[tf.Variable, tf.Variable, tf.Variable]:
    curr_input, fw_final, bw_final = input, None, None

    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('bigru_{}'.format(i)) as scope:
            if dropout_function is not None and i is not 0:
                curr_input = dropout_function(curr_input)

            fw_cell = GRUCell(layer_size)
            bw_cell = GRUCell(layer_size)
            (fw_out, bw_out), (fw_final, bw_final) = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=curr_input,
                                                                               dtype=tf.float32,
                                                                               sequence_length=input_length,
                                                                               scope=scope,
                                                                               parallel_iterations=parallel_iterations,
                                                                               swap_memory=True)
            curr_input = tf.concat([fw_out, bw_out], axis=2)

    return curr_input, fw_final, bw_final

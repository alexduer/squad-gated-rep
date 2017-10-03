from typing import Tuple

import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from models.common.match_rnn_cell import MatchRNNCell, MatchRNNCellV2


def rnet_matching_layer(layer_size: int, att_size: int, par_vecs: tf.Variable, qu_vecs: tf.Variable,
                        par_num_words: tf.Variable, parallel_iterations: int=64) -> tf.Variable:
    with tf.variable_scope('alignment_par_qu') as scope:
        with tf.variable_scope('fw/match_rnn_cell/attention'):
            fw_cell = MatchRNNCell(GRUCell(layer_size), qu_vecs, att_size)

        with tf.variable_scope('bw/match_rnn_cell/attention'):
            bw_cell = MatchRNNCell(GRUCell(layer_size), qu_vecs, att_size)

        (fw_out, bw_out), (_, _) = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=par_vecs,
                                                             dtype=tf.float32, sequence_length=par_num_words,
                                                             scope=scope, parallel_iterations=parallel_iterations,
                                                             swap_memory=True)
        match_par_qu_out = tf.concat([fw_out, bw_out], axis=2)

    return match_par_qu_out


def rnet_matching_layer_unidirectional(layer_size: int, att_size: int, par_vecs: tf.Variable, qu_vecs: tf.Variable,
                                       par_num_words: tf.Variable, parallel_iterations: int=64) -> tf.Variable:
    with tf.variable_scope('alignment_par_qu') as scope:
        with tf.variable_scope('fw/match_rnn_cell/attention'):
            rnn_cell = MatchRNNCell(GRUCell(layer_size), qu_vecs, att_size)

        output, _ = dynamic_rnn(rnn_cell, inputs=par_vecs, dtype=tf.float32, sequence_length=par_num_words, scope=scope,
                                parallel_iterations=parallel_iterations, swap_memory=True)

    return output


def rnet_self_matching_layer(layer_size: int, att_size: int, par_vecs: tf.Variable,
                             par_num_words: tf.Variable, parallel_iterations: int=64) -> tf.Variable:
    with tf.variable_scope('alignment_self') as scope:
        with tf.variable_scope('fw/match_rnn_cell/attention'):
            fw_cell = MatchRNNCell(GRUCell(layer_size), par_vecs, att_size, use_state_for_att=False)

        with tf.variable_scope('bw/match_rnn_cell/attention'):
            bw_cell = MatchRNNCell(GRUCell(layer_size), par_vecs, att_size, use_state_for_att=False)

        (fw_out, bw_out), (_, _) = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=par_vecs,
                                                             dtype=tf.float32, sequence_length=par_num_words,
                                                             scope=scope, parallel_iterations=parallel_iterations,
                                                             swap_memory=True)
        match_self_out = tf.concat([fw_out, bw_out], axis=2)

    return match_self_out


def rnet_self_matching_layer_real(layer_size: int, att_size: int, par_vecs: tf.Variable,
                                  par_num_words: tf.Variable, parallel_iterations: int=64) -> tf.Variable:
    with tf.variable_scope('alignment_self') as scope:
        WP = tf.get_variable('WP', [2*layer_size, att_size])
        WPtilde = tf.get_variable('WPtilde', [2*layer_size, att_size])
        v = tf.get_variable('v', [att_size])
        att_match_input = tf.einsum('ijk,kl->ijl', par_vecs, WPtilde)

        with tf.variable_scope('fw/match_rnn_cell/attention'):
            fw_cell = MatchRNNCellV2(GRUCell(layer_size), WP, v, par_vecs, att_match_input)

        with tf.variable_scope('bw/match_rnn_cell/attention'):
            bw_cell = MatchRNNCellV2(GRUCell(layer_size), WP, v, par_vecs, att_match_input)

        (fw_out, bw_out), (_, _) = bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=par_vecs,
                                                             dtype=tf.float32, sequence_length=par_num_words,
                                                             scope=scope, parallel_iterations=parallel_iterations,
                                                             swap_memory=True)
        match_self_out = tf.concat([fw_out, bw_out], axis=2)

    return match_self_out


def _gate_input(input: tf.Variable) -> tf.Variable:
    agg_dim = int(input.get_shape()[2])
    W = tf.get_variable('W', [agg_dim, agg_dim])
    g_t = tf.sigmoid(tf.einsum('ijk,kl->ijl', input, W))
    return tf.multiply(g_t, input)


def rnet_prediction(par_vecs: tf.Variable, qu_vecs: tf.Variable, layer_size: int, att_size:int) -> tf.Variable:
    h_0 = _prediction_init_state(qu_vecs, layer_size)
    h_1, pred_start = _prediction_layer(par_vecs, layer_size, att_size, h_0)
    _, pred_end = _prediction_layer(par_vecs, layer_size, att_size, h_1, True)
    prediction_log_probs = tf.stack([pred_start, pred_end], 1)
    return prediction_log_probs


def _prediction_init_state(qu_vecs: tf.Variable, att_size: int) -> tf.Variable:
    qu_vec_dim = int(qu_vecs.get_shape()[2])

    with tf.variable_scope('prediction_init'):
        v = tf.get_variable('v', [att_size], tf.float32)
        WQu = tf.get_variable('WQu', [qu_vec_dim, att_size], tf.float32)
        WQvVQr = tf.get_variable('WQvVQr', [1, 1, att_size], tf.float32)

    # do calculations
    # s^t_j
    # [batch_size, num_words, layer_size]
    att_term = tf.add(tf.einsum('ijk,kl->ijl', qu_vecs, WQu), WQvVQr)
    # [batch_size, num_words]
    s = tf.einsum('ijl,l->ij', tf.nn.tanh(att_term), v)

    # a^t_i
    a = tf.nn.softmax(s, dim=1)

    # r^Q (is the equivalent of c in prediction_layer)
    # [batch_size, layer_size]
    r = tf.reduce_sum(tf.multiply(tf.expand_dims(a, axis=2), qu_vecs), axis=1)

    return r


def _prediction_layer(aggregation: tf.Variable, layer_size: int, att_size:int, prev_rnn_state: tf.Variable = None,
                      reuse: bool = False) -> Tuple[tf.Variable, tf.Variable]:
    batch_size = int(aggregation.get_shape()[0])
    agg_dim = int(aggregation.get_shape()[2])
    rnn_size = int(prev_rnn_state.get_shape()[1])

    # actual logic
    with tf.variable_scope('prediction_static', reuse=reuse):
        WPh = tf.get_variable('WPh', [agg_dim, att_size], tf.float32)
        Wah = tf.get_variable('Wah', [rnn_size, att_size], tf.float32)
        v = tf.get_variable('v', [att_size], tf.float32)

    # do calculations
    # s^t_j
    # [batch_size, num_words, att_size]
    att_term = tf.einsum('ijk,kl->ijl', aggregation, WPh)
    # [batch_size, att_size]
    rnn_term = tf.matmul(prev_rnn_state, Wah)
    # [batch_size, num_words, att_size]
    term = tf.add(att_term, tf.expand_dims(rnn_term, axis=1))

    # [batch_size, num_words]
    s = tf.einsum('ijl,l->ij', tf.tanh(term), v)

    # a^t_i
    a = tf.nn.softmax(s, dim=1)

    # c_t
    c = tf.reduce_sum(tf.multiply(tf.expand_dims(a, axis=2), aggregation), axis=1)

    # next h^a_t
    with tf.variable_scope('prediction_static_rnn', reuse=reuse) as scope:
        rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_size, reuse=reuse)
        _, next_hat = tf.nn.static_rnn(rnn_cell, [c], initial_state=prev_rnn_state,
                                       sequence_length=tf.ones([batch_size], tf.int32), scope=scope,
                                       dtype=tf.float32)

    return next_hat, tf.log(a)

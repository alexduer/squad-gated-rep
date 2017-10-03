import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear


class MatchRNNCell(RNNCell):
    def __init__(self,
                 base_cell: RNNCell,
                 match_input: tf.Variable,
                 num_units: int,
                 reuse=None, use_state_for_att=True, use_att_bias=False):
        super(MatchRNNCell, self).__init__(_reuse=reuse)
        self._base_cell = base_cell
        self._match_input = match_input
        self._match_size = int(self._match_input.get_shape()[2])
        self._num_units = num_units
        self._use_state_for_att = use_state_for_att
        self._use_att_bias = use_att_bias

        with tf.variable_scope("match_input"):
            match_weights = tf.get_variable('match_weights', [self._match_size, self._num_units])
            # [batch_size, num_match_elems, layer_size]
            self._att_match_in = tf.einsum('ijk,kl->ijl', self._match_input, match_weights)

    @property
    def state_size(self):
        return self._base_cell.state_size

    @property
    def output_size(self):
        return self._base_cell.output_size

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        # inputs: [batch_size, in_size]
        # state:  [batch_size, output_size OR state_size]
        with tf.variable_scope("attention"):
            with tf.variable_scope("main_input"):
                # [batch_size, 1, layer_size]
                att_main_in = tf.expand_dims(_linear([inputs], self._num_units, self._use_att_bias), axis=1)

            with tf.variable_scope("state_input"):
                # [batch_size, 1, layer_size]
                att_state_in = tf.expand_dims(_linear([state], self._num_units, False), axis=1)

            with tf.variable_scope("s"):
                att_vec = tf.get_variable('att_vec', [self._num_units])
                # [batch_size, num_match_elems, layer_size]
                if self._use_state_for_att:
                    raw_in = tf.add(tf.add(att_main_in, att_state_in), self._att_match_in)
                else:
                    raw_in = tf.add(att_main_in, self._att_match_in)
                # [batch_size, num_match_elems, 1]
                s = tf.einsum('ijk,k->ij', tf.nn.tanh(raw_in), att_vec)

            # [batch_size, num_match_elems]
            a = tf.nn.softmax(s, dim=1)
            # [batch_size, match_size]
            c = tf.reduce_sum(tf.multiply(tf.expand_dims(a, axis=2), self._match_input), axis=1)

        raw_rnn_inputs = tf.concat([inputs, c], axis=1)

        with tf.variable_scope("output_gate"):
            rnn_input_size = int(raw_rnn_inputs.get_shape()[1])
            rnn_input_gate = tf.sigmoid(_linear([raw_rnn_inputs], rnn_input_size, False))
            rnn_inputs = tf.multiply(raw_rnn_inputs, rnn_input_gate)

        new_h, new_h = self._base_cell.call(inputs=rnn_inputs, state=state)
        return new_h, new_h


class MatchRNNCellV2(RNNCell):
    def __init__(self,
                 base_cell: RNNCell,
                 WP: tf.Variable,
                 v: tf.Variable,
                 match_input: tf.Variable,
                 att_match_input: tf.Variable,
                 reuse=None):
        super(MatchRNNCellV2, self).__init__(_reuse=reuse)
        self._base_cell = base_cell
        self._WP = WP
        self._v = v
        self._match_input = match_input
        self._att_match_input = att_match_input
        self._match_size = int(self._match_input.get_shape()[2])

    @property
    def state_size(self):
        return self._base_cell.state_size

    @property
    def output_size(self):
        return self._base_cell.output_size

    # noinspection PyMethodOverriding
    def call(self, inputs, state):
        # inputs: [batch_size, in_size]
        # state:  [batch_size, output_size OR state_size]
        with tf.variable_scope("attention"):
            with tf.variable_scope("main_input"):
                # [batch_size, 1, att_size]
                att_main_in = tf.expand_dims(tf.einsum('ij,jk->ik', inputs, self._WP), axis=1)

            with tf.variable_scope("s"):
                # [batch_size, num_match_elems, att_size]
                raw_in = tf.add(att_main_in, self._att_match_input)
                # [batch_size, num_match_elems, 1]
                s = tf.einsum('ijk,k->ij', tf.nn.tanh(raw_in), self._v)

            # [batch_size, num_match_elems]
            a = tf.nn.softmax(s, dim=1)
            # [batch_size, match_size]
            c = tf.reduce_sum(tf.multiply(tf.expand_dims(a, axis=2), self._match_input), axis=1)

        raw_rnn_inputs = tf.concat([inputs, c], axis=1)

        with tf.variable_scope("pre_input_gate"):
            rnn_input_size = int(raw_rnn_inputs.get_shape()[1])
            rnn_input_gate = tf.sigmoid(_linear([raw_rnn_inputs], rnn_input_size, False))
            rnn_inputs = tf.multiply(raw_rnn_inputs, rnn_input_gate)

        new_h, new_h = self._base_cell.call(inputs=rnn_inputs, state=state)
        return new_h, new_h

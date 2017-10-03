import tensorflow as tf
import numpy as np


def loss_neg_log_prob(prediction_logits: tf.Variable, labels: tf.Variable) -> tf.Variable:
    print('labels', labels.get_shape())
    batch_size = int(prediction_logits.get_shape()[0])

    prediction_list = tf.unstack(tf.log(tf.nn.softmax(prediction_logits, dim=2)), axis=1)
    pred_start = prediction_list[0]
    pred_end = prediction_list[1]

    label_list = tf.unstack(labels, axis=1)
    label_start = label_list[0]
    label_end = label_list[1]

    negative_log_prob_start = tf.div(_extract_vec_values(pred_start, label_start),
                                     tf.constant(-batch_size, dtype=tf.float32))
    negative_log_prob_end = tf.div(_extract_vec_values(pred_end, label_end),
                                   tf.constant(-batch_size, dtype=tf.float32))

    return tf.add(negative_log_prob_start, negative_log_prob_end) / tf.constant(2, dtype=tf.float32, shape=[])


def _extract_vec_values(batched_vecs: tf.Variable, batched_indices: tf.Variable) -> tf.Variable:
    batch_size = int(batched_vecs.get_shape()[0])

    batch_index_range = tf.constant(np.array([n for n in range(batch_size)], dtype=np.int32), dtype=tf.int32)
    full_batched_indices = tf.stack([batch_index_range, batched_indices], axis=1)
    return tf.gather_nd(batched_vecs, full_batched_indices)
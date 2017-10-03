import json
import os
import random
import time

import tensorflow as tf

from common.data.dataset.load import load_dataset
from common.data.download import prepare_data
from models import create_model

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

flags.DEFINE_string('mode', 'training', 'one of: training, inference')
flags.DEFINE_string('model', 'Rnet', 'name of the model to run')
flags.DEFINE_integer('random_seed', int(round(time.time() * 1000)), 'random seed to make model training replicable')
flags.DEFINE_boolean('filter_short_questions', True, 'filter questions with less than 4 characters')
flags.DEFINE_boolean('save_meta_graph', False, 'save Tensorflow meta graph together with weights')


# actual model params
flags.DEFINE_string('layer_size', 50, 'size for most neural network layers')
flags.DEFINE_string('att_size', 50, 'size attention layers')
flags.DEFINE_boolean('load_embeddings_from_file', False, 'load char embeddings from single chracter words of word embeddings')
flags.DEFINE_string('char_embedding_size', 50, 'size of RNN layer that creates a single vector out of all characters of a word')
flags.DEFINE_string('char_rnn_size', 50, 'size of RNN layer that creates a single vector out of all characters of a word')
flags.DEFINE_float('dropout', 0.8, 'dropout keep probability used during training')

flags.DEFINE_integer('max_par_length', 600, 'maximum number of tokens in paragraph to keep, longer paragraphs will be truncated')
flags.DEFINE_integer('max_qu_length', 39, 'maximum number of tokens in question to keep, longer questions will be truncated')
flags.DEFINE_integer('max_char_length', 29, 'maximum number of characters per word to keep, longer words will be truncated')
flags.DEFINE_integer('rnn_parallelity', 32, 'maximum parallelity to use for RNNs')
flags.DEFINE_integer('iter_bucket_size', 256, 'bucket size to group ')

# input training params
flags.DEFINE_integer('batch_size', 32, 'batch size to use for model')
flags.DEFINE_integer('apply_grads_interval', 1, 'apply gradient updates every n batches')
flags.DEFINE_integer('max_epochs', 1000, 'maximum number of epochs to train for')
flags.DEFINE_integer('patience', 1e6, 'maximum number of times which learning rate is halved')
flags.DEFINE_string('lr_reduction_criterion', 'F1', 'benchmark criterion for learning rate reduction, valid values: EM, F1')

flags.DEFINE_string('optimizer', 'Adam', 'optimizer to use, valid values: Adam, Adadelta')
flags.DEFINE_float('opt_lr', 1e-3, 'initial learning rate for optimizer')
flags.DEFINE_float('opt_epsilon', 1e-6, 'epsilon value to use for optimizer')
flags.DEFINE_float('opt_adam_beta1', 0.9,'beta1 value to use for Adam optimizer')
flags.DEFINE_float('opt_adam_beta2', 0.999,'beta2 value to use for Adam optimizer')
flags.DEFINE_float('opt_adadelta_rho', 0.95,'rho value to use for Adadelta optimizer')

# folders etc
flags.DEFINE_string('data_dir', MAIN_DIR + '/files/data', 'directory where required data files (e.g. word embeddings) are downloaded to and saved')
flags.DEFINE_string('cache_dir', MAIN_DIR + '/files/cache', 'directory where long calculations like tokenization are cached')
flags.DEFINE_string('weights_dir', '/tmp', 'directory to load weights from and save new weights to')
flags.DEFINE_string('results_file', '/tmp/results.json', 'file to save inference results to')
flags.DEFINE_string('dataset', MAIN_DIR + '/files/data/train-v1.1.json', 'which dataset to use for training or inference')
flags.DEFINE_string('valid_dataset', MAIN_DIR + '/files/data/dev-v1.1.json', 'dataset to use for validation during training')
flags.DEFINE_string('placement', 'gpu', 'placement of Tensorflow computations, valid values: gpu, cpu')


def main(_):
    config = flags.FLAGS.__flags.copy()

    tf.set_random_seed(config['random_seed'])
    random.seed(config['random_seed'])

    prepare_data(config['data_dir'])
    dataset = load_dataset(config['dataset'], config['data_dir'], config['cache_dir'])

    if config['mode'] == 'training':
        print('starting in training mode ...')
        valid_dataset = load_dataset(config['valid_dataset'], config['data_dir'], config['cache_dir'])
        model = create_model(config['model'], config, [dataset, valid_dataset])
        model.train(dataset, valid_dataset, config['max_epochs'], config['patience'])
        print('finished training, exiting ...')
    elif config['mode'] == 'inference':
        print('starting in inference mode ...')
        model = create_model(config['model'], config, [dataset])
        inference_result = model.infer(dataset)
        with open(config['results_file'], 'w') as file:
            json.dump(inference_result, file)
        print('finished inference, dumped results to {}'.format(config['results_file']))
    else:
        raise Exception('illegal mode value: {}, only training and inference are allowed'.format(config['mode']))

    print('exiting ...')


if __name__ == '__main__':
    tf.app.run()

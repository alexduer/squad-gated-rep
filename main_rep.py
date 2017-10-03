import os
import sys
from subprocess import Popen

from common.util.time import now

for i in range(12):
    print('[{}] reproducing experiment {}'.format(now(), i))

    if len(sys.argv) <= 1:
        save_path = '/tmp'
    else:
        save_path = sys.argv[1]

    out_path = os.path.join(save_path, 'model_{}'.format(i))
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    log_file = os.path.join(out_path, 'log_{}.txt'.format(i))

    ARGS = [
        '--model=RnetRep{}'.format(i),
        '--layer_size=75',
        '--att_size=75',
        '--char_embedding_size=75',
        '--char_rnn_size=75',
        '--dropout=0.8',
        '--max_par_length=767',
        '--max_qu_length=60',
        '--max_char_length=37',
        '--batch_size=16',
        '--apply_grads_interval=2',
        '--patience=10',
        '--optimizer=Adadelta',
        '--opt_lr=1.0',
        '--weights_dir={}'.format(out_path),
        '--lr_reduction_criterion=EM',
    ]

    print('[{}] starting training script'.format(now()))
    command = ' '.join(['python3', '-u', 'main.py', *ARGS, '>', log_file, '2>&1'])
    print('[{}] running command: {}'.format(now(), command))
    process = Popen(command, shell=True)
    print('[{} | {}] waiting for training script to finish ...'.format(now(), process.pid))
    process.wait()

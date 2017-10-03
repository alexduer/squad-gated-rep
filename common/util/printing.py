import math
from typing import Union, List

from common.util.time import now

import tensorflow as tf


def print_var(name, var: Union[tf.Variable, List[tf.Variable]]) -> None:
    if isinstance(var, list):
        print('{}: {}x{}'.format(name, len(var), var[0].get_shape()))
    else:
        print('{}: {}'.format(name, var.get_shape()))


def ordinal(n: int):
    return "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])


def print_epoch_summary(epoch_id: int, em_score: float, f1_score: float) -> None:
    print('===============   [{} | {}]   ==============='.format(now(), epoch_id))
    print('accuracy (EM): ', em_score)
    print('f1: ', f1_score)
    print('==================================================================')


def print_batch_summary(epoch_id: int, em_score: float, f1_score: float) -> None:
    print('[{} | {}] accuracy (EM): {}, f1: {}'.format(now(), epoch_id, em_score, f1_score))
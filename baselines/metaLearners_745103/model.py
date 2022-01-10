import time
t1 = time.time()

from typing import Dict
import json
import os
from copy import deepcopy
try:
    import torch
except:
    os.system('conda install --yes pytorch torchvision cudatoolkit=10.1 -c pytorch')

import tensorflow as tf

from src.utils import get_logger
LOGGER = get_logger('GLOBAL')

os.system('pip list')

gpus = tf.config.experimental.list_physical_devices('GPU')
LOGGER.info('gpus log', gpus)
LOGGER.info('before everything nvidia-smi')
os.system('nvidia-smi')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# set the visible devices of tf
# since some random split is used, its better we load data using gpu
# force the tf program run on cuda:0
# not working because in the environment, the tf will be initialized first and then our code
if isinstance(gpus, list) and len(gpus) > 3:
    tf.config.experimental.set_visible_devices(devices=gpus[0:3], device_type='GPU')

current_path = os.path.abspath(__file__)
config_file_path = os.path.abspath(os.path.join(os.path.dirname(current_path), 'config.json'))
import json
config = json.load(open(config_file_path, 'r'))

from src.meta.multiprocess import single
from src.meta.proto import prototype_multi

if config['epochs'] == 'auto':
    config['epochs'] = prototype_multi.MAX_LOCAL_EPOCH

def apply_device(conf: Dict, device, global_id, **kwargs):
    conf['device'] = 'cuda:{}'.format(device)
    conf['global_id'] = global_id
    conf.update(kwargs)
    return conf

single.GLOBAL_CONFIG = {
    #'modules': [prototype_multi, prototype_multi, prototype_multi, prototype_multi],
    'modules': prototype_multi,
    'hp': apply_device(deepcopy(config), 1, 1, backbone='resnet50'),
    'device': 'cuda:0',
    'begin_time_stamp': t1
}

LOGGER.info(single.GLOBAL_CONFIG)

MyMetaLearner = single.MyMetaLearner
MyLearner = single.MyLearner
MyPredictor = single.MyMultiPredictor

t2 = time.time()
LOGGER.info('time used for installing package, set-up gpu', t2 - t1)

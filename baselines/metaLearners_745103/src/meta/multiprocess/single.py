# single version

from copy import deepcopy
import time
import os
import traceback
import tensorflow as tf
from queue import Queue
import numpy as np

import torch
import threading
from src.utils.utils import timer
from metadl.api.api import MetaLearner, Learner, Predictor
from itertools import cycle
from src.utils.utils import process_task_batch, to_torch, get_base_class_number, PipeSync, ThreadPipe, QueuePipeWrapper
from src.utils import logger
from src.utils.gpu import GPUManager

LOGGER = logger.get_logger('main-single')

GLOBAL_CONFIG = {}
STATUS = {}

def predict_single(learner, episode_data, total_task=200, res='cpu', kargs={}):
    device = torch.device(res)
    learner.to(device)
    result = []
    for i in range(total_task):
        supp_x, supp_y, quer_x = episode_data[i]
        res = learner.fit(supp_x, supp_y, quer_x, device=device)
        result.extend(res.tolist())
    learner.to(torch.device('cpu'))
    return {'res': result, **kargs}

def run_exp_single(module, hp, train, valid, clsnum):
    try:
        mlearner = module(hp)
        mlearner.meta_fit(train, valid, clsnum)
    except:
        LOGGER.info('exp', hp['global_id'], 'terminated with the following error')
        traceback.print_exc()
        LOGGER.info('will exit the experiments')

class MyMetaLearner(MetaLearner):
    def __init__(self) -> None:
        self.timer = timer().initialize(time_begin=GLOBAL_CONFIG['begin_time_stamp'], time_limit=60 * 110)
        self.__dict__.update(GLOBAL_CONFIG)
        self.hp['time_begin'] = GLOBAL_CONFIG['begin_time_stamp']
        LOGGER.info('initialization done')
        
    def meta_fit(self, meta_dataset_generator):

        manager = GPUManager()
        LOGGER.debug('My PID: %s' % os.getpid())

        finished = False

        def display_gpu():
            while not finished:
                time.sleep(0.5)
                manager.print_processes(prin=LOGGER.info)

        th = threading.Thread(target=display_gpu)
        th.daemon = True
        th.start()

        self.timer.begin('main training')
        self.timer.begin('build main proc pipeline')
        clsnum = get_base_class_number(meta_dataset_generator)
        LOGGER.info('base class number detected', clsnum)

        self.timer.begin('prepare dataset')
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline.batch(1)
        meta_train_generator = cycle(iter(meta_train_dataset))
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline.batch(1)
        meta_valid_generator = cycle(iter(meta_valid_dataset))
        self.timer.end('prepare dataset')

        # get the most suitable GPU to run the code
        try:
            gpu = manager.get_gpus(remain_memory=3000)[0].idx
            self.hp['device'] = 'cuda:{}'.format(gpu)
        except:
            self.hp['device'] = 'cpu'
        
        LOGGER.info('available device', self.hp['device'])
        device = torch.device(self.hp['device'])
        devc = torch.device('cpu')

        self.timer.begin('cal stat')
        global STATUS
        # before process dataset, we need to query one batch to calculate the statics of input
        image_list = []
        datas = []
        for i in range(10):
            batch = next(meta_train_generator)
            supp, quer = process_task_batch(batch, device, with_origin_label=True)
            datas.append([supp, quer])
            image_list.append(supp[0])
            image_list.append(quer[0])
        # image_list: B H W C
        image_list = torch.cat(image_list, dim=0)
        stds, means = torch.std_mean(image_list, dim=(0, 2, 3), keepdim=True)
        STATUS['mean'] = means
        STATUS['std'] = stds + 1e-6
        self.timer.end('cal stat')
        LOGGER.info('cal stat spent time', self.timer.query_time_by_name('cal stat'))
        LOGGER.info('mean', STATUS['mean'].tolist())
        LOGGER.info('std', STATUS['std'].tolist())

        def process_supp_quer(supp, quer):
            supp[0] = (supp[0] - STATUS['mean']) / STATUS['std']
            quer[0] = (quer[0] - STATUS['mean']) / STATUS['std']
            return supp, quer

        # data processing thread
        train_queue, valid_queue = Queue(32 * 10), Queue(200)

        finished_data = False

        def load_dataset():
            while not finished_data:
                for i in range(32 * 10):
                    # load train data
                    if len(datas) > 0:
                        supp, quer = datas.pop()
                    else:
                        supp, quer = process_task_batch(next(meta_train_generator), device=device, with_origin_label=True)
                    supp, quer = process_supp_quer(supp, quer)
                    data = self.modules.process_data(supp, quer, True, self.hp)
                    # CAUTION: may lock the thread!
                    train_queue.put(data)
                    if finished_data:
                        break
                    time.sleep(0.01)

                if finished_data:
                    break

                for i in range(200):
                    supp, quer = process_task_batch(next(meta_valid_generator), device=device, with_origin_label=False)
                    supp, quer = process_supp_quer(supp, quer)
                    data = self.modules.process_data(supp, quer, False, self.hp)
                    valid_queue.put(data)
                    if finished_data:
                        break
                    time.sleep(0.01)

                if finished_data:
                    break
        
        th1 = threading.Thread(target=load_dataset)
        th1.daemon = True
        th1.start()

        LOGGER.info('prepare dataset', self.timer.query_time_by_name('prepare dataset'))

        # log the nvidia info
        LOGGER.info('before spawning tasks nvidia-smi')
        os.system('nvidia-smi')

        run_exp_single(self.modules.MyMetaLearner, self.hp, QueuePipeWrapper(train_queue), QueuePipeWrapper(valid_queue), clsnum)
        finished_data = True
        if train_queue.qsize() > 0:
            train_queue.get()
        if valid_queue.qsize() > 0:
            valid_queue.get()

        self.meta_learner = self.modules.load_model(self.hp)

        finished = True
        th.join()

        return MyLearner(self.meta_learner, timers=self.timer, status=STATUS)

GPU_MANAGER = GPUManager()

# change the logic to put all the supp img, lab to MyPredictor
# there is a method that we can start the process as long as the load() is exececuted
class MyLearner(Learner):
    def __init__(self, meta_learners=None, timers=None, status=None) -> None:
        self.__dict__.update(GLOBAL_CONFIG)
        self.timer = timers
        self.learner = meta_learners
        self.loaded = False
        self.status = status
        self.epoch = 0
    
    def fit(self, dataset_train):

        self.timer.begin('predict')
        finished = False

        def display_gpu():
            while not finished:
                time.sleep(0.5)
                GPU_MANAGER.print_processes(prin=LOGGER.info)

        th = threading.Thread(target=display_gpu)
        th.daemon = True
        th.start()
        img, lab = None, None
        if (self.epoch % 50 == 0) and self.epoch > 0:
            LOGGER.info('mean testing speed', self.timer.query_time_by_name('predict'), 'estimated max time left for one epoch', self.timer.time_left() / max((600 - self.epoch), 1))
        self.epoch += 1
        for idx, (image, label) in enumerate(dataset_train):
            img = to_torch(image).permute(0, 3, 1, 2)
            lab = to_torch(label, dtype=torch.long)
        finished = True
        th.join()
        return MyMultiPredictor(self.learner, [img, lab], self.timer)

    def save(self, path_to_save):
        t1 = time.time()
        self.learner.save(path_to_save)
        open(os.path.join(path_to_save, 'time_state.txt'), 'w').write(str(self.timer.time_begin))
        torch.save(self.status, os.path.join(path_to_save, 'status.pt'))
        t2 = time.time()
        LOGGER.info('save time', t2 - t1)
            
    def load(self, path_to_model):
        if not self.loaded:
            t1 = time.time()
            time_begin = eval(open(os.path.join(path_to_model, 'time_state.txt')).read().strip())
            self.timer = timer().initialize(time_begin=time_begin, time_limit=60 * 110)
            LOGGER.info('time left for test', self.timer.time_left())
            module = self.modules
            # get the most suitable GPU to run the code
            try:
                gpu = GPU_MANAGER.get_gpus(remain_memory=3000)[0].idx
                self.hp['device'] = 'cuda:{}'.format(gpu)
            except:
                self.hp['device'] = 'cpu'
            
            LOGGER.info('available device', self.hp['device'])
            self.status = torch.load(os.path.join(path_to_model, 'status.pt'), map_location=self.hp['device'])
            global STATUS
            STATUS = self.status
            learner = module.MyMultiManager(None, self.hp)
            learner.load(path_to_model, device=self.hp['device'])
            self.learner = learner
            self.loaded = True

class MyMultiPredictor(Predictor):
    def __init__(self, learner, supp, timers) -> None:
        self.__dict__.update(GLOBAL_CONFIG)
        self.learner = learner
        self.supp = supp
        self.timer = timers
    
    def predict(self, dataset_test):
        finished = False

        def display_gpu():
            while not finished:
                time.sleep(0.5)
                GPU_MANAGER.print_processes(prin=LOGGER.info)

        th = threading.Thread(target=display_gpu)
        th.daemon = True
        th.start()

        for image in dataset_test:
            image = image[0]
            # 95 * 28 * 28 * 3
            image = to_torch(image).permute(0, 3, 1, 2).to(STATUS['mean'].device)
            self.supp[0] = self.supp[0].to(STATUS['mean'].device)
            self.supp[1] = self.supp[1].to(STATUS['mean'].device)
            image = (image - STATUS['mean']) / STATUS['std']
            self.supp[0] = (self.supp[0] - STATUS['mean']) / STATUS['std']
            finished = True
            th.join()
            x = self.learner.fit(self.supp[0], self.supp[1], image, device=self.learner.config['device'])
            self.timer.end('predict')
            return x

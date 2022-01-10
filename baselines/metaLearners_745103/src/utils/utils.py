from copy import deepcopy
import os
import threading
import numpy as np
import random
import torch
import tensorflow as tf
import time
from torchvision import transforms
from itertools import cycle

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)

def to_torch(source, dtype=None) -> torch.Tensor:
    if dtype is None:
        return torch.from_numpy(source.numpy())
    return torch.from_numpy(source.numpy()).to(dtype)

def to_BCHW(source: torch.Tensor):
    return source.permute(0, 3, 1, 2)

def process_task_batch_numpy(batch, with_origin_label=False):
    # supp  image: batch[0][0]: batch * (WAY * SHOT) * H * W * C
    # supp  label: batch[0][1]: batch * (WAY * SHOT)
    # query image: batch[0][3]: batch * (WAY * QUER) * H * W * C
    # query label: batch[0][4]: batch * (WAY * QUER)

    supp = [batch[0][0].numpy().transpose((0, 1, 4, 2, 3)), batch[0][1].numpy()]
    quer = [batch[0][3].numpy().transpose((0, 1, 4, 2, 3)), batch[0][4].numpy()]
    if with_origin_label:
        supp += [batch[0][2].numpy()]
        quer += [batch[0][5].numpy()]
    
    return supp, quer


def process_task_batch(batch, device=torch.device('cuda:0'), with_origin_label=False, data_augmentor=None):
    # supp  image: batch[0][0]: 1 * (WAY * SHOT) * H * W * C
    # supp  label: batch[0][1]: 1 * (WAY * SHOT)
    # query image: batch[0][3]: 1 * (WAY * QUER) * H * W * C
    # query label: batch[0][4]: 1 * (WAY * QUER)

    supp = [to_torch(batch[0][0])[0].permute(0, 3, 1, 2), to_torch(batch[0][1], dtype=torch.long)[0].to(device)]
    query = [to_torch(batch[0][3])[0].permute(0, 3, 1, 2), to_torch(batch[0][4], dtype=torch.long)[0].to(device)]
    if data_augmentor is not None:
        supp[0] = data_augmentor(supp[0])
        query[0] = data_augmentor(query[0])
    supp[0] = supp[0].to(device)
    query[0] = query[0].to(device)
    if with_origin_label:
        supp += [to_torch(batch[0][2], dtype=torch.long)[0].to(device)]
        query += [to_torch(batch[0][5], dtype=torch.long)[0].to(device)]
    
    return supp, query

def mean(x):
    return sum(x) / len(x)

class DataArgumentor():
    def __init__(self, device=torch.device('cuda:0'), mean=None, std=None):
        self.device = device
        self.transforms_list = [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30)
        ]
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.3, 1.0))
        ])
        self.mean = None
        self.std = None
        self.normalizer = None
        self.updateMeanStd(mean, std)
    
    def updateMeanStd(self, mean, std):
        self.mean = mean
        self.std = std
        if mean is not None and std is not None:
            self.normalizer = transforms.Normalize(mean, std)
    
    def __call__(self, src):
        src = self.augmentation(src)
        if self.normalizer is not None:
            src = self.normalizer(src)
        return src

class queue_sync():
    def __init__(self, dataset) -> None:
        self.meta_dataset = cycle(iter(dataset))
    
    def get(self):
        data = next(self.meta_dataset)
        return process_task_batch(data, device=torch.device('cpu'), with_origin_label=True)

def build_queue_sync(meta_dataset_generator):
    return queue_sync(meta_dataset_generator.meta_train_pipeline.batch(1)), queue_sync(meta_dataset_generator.meta_valid_pipeline.batch(1))

class QueuePipeWrapper():
    def __init__(self, queue) -> None:
        super().__init__()
        self.queue = queue
    
    def send(self, x):
        return True
    
    def recv(self):
        return self.queue.get()

class ThreadPipe():
    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.data = None
    
    def send(self, x):
        while self.data is not None:
            time.sleep(0.01)
        self.lock.acquire()
        self.data = x
        self.lock.release()
        return True
    
    def recv(self):
        while self.data is None:
            time.sleep(0.01)
        self.lock.acquire()
        x = self.data
        self.data = None
        self.lock.release()
        return x

class PipeSync():
    def __init__(self, func, **kwargs) -> None:
        self.func = func
        self.kwargs = kwargs
    
    def recv(self):
        return self.func(**self.kwargs)
    
    def send(self, x):
        return True

class timer():
    def initialize(self, time_begin='auto', time_limit=60 * 100):
        self.time_limit = time_limit
        self.time_begin = time.time() if time_begin == 'auto' else time_begin
        self.time_list = [self.time_begin]
        self.named_time = {}
        '''
        'name' : {
            'time_begin': xxx,
            'time_period': [],
        }
        '''
        return self

    def anchor(self, name=None, end=None):
        self.time_list.append(time.time())
        if name is not None:
            if name in self.named_time:
                if end:
                    assert self.named_time[name]['time_begin'] is not None
                    self.named_time[name]['time_period'].append(self.time_list[-1] - self.named_time[name]['time_begin'])
                else:
                    self.named_time[name]['time_begin'] = self.time_list[-1]
            else:
                assert end == False
                self.named_time[name] = {
                    'time_begin': self.time_list[-1],
                    'time_period': []
                }
        return self.time_list[-1] - self.time_list[-2]

    def query_time_by_name(self, name, method=mean, default=50):
        if name not in self.named_time or self.named_time[name]['time_period'] == []:
            return default
        times = self.named_time[name]['time_period']
        return method(times)

    def time_left(self):
        return self.time_limit - time.time() + self.time_begin
    
    def begin(self, name):
        self.anchor(name, end=False)
    
    def end(self, name):
        self.anchor(name, end=True)
        return self.named_time[name]['time_period'][-1]

def get_base_class_number(*args):
    return 70

def get_root(path = None):
    root = os.path.abspath(os.path.join(os.path.dirname(__name__), '../../'))
    if path is None:
        return root
    return os.path.join(root, path)

def aggressive_decoding(out):
    if isinstance(out, torch.Tensor):
        out = np.array(out.detach().cpu())
    cur_num_per_cls = [0, 0, 0, 0, 0]
    num_per_cls = out.shape[0]//out.shape[1]
    # print(num_per_cls)
    result = np.zeros(out.shape)
    while sum(cur_num_per_cls) < out.shape[0]:
        max_per_line = np.max(out, axis=1)
        max_per_line_index = np.argmax(out, axis=1)
        cur_max_index = np.argsort(max_per_line)[-1]
        cur_max = max_per_line[cur_max_index]
        cur_max_cls = max_per_line_index[cur_max_index]
        if cur_num_per_cls[cur_max_cls] < num_per_cls:
            cur_num_per_cls[cur_max_cls] += 1
            result[cur_max_index, cur_max_cls] = 1
            out[cur_max_index] = out[cur_max_index] - out[cur_max_index]
        else:
            out[cur_max_index, cur_max_cls] = 0
    return result


def viterbi_decoding(out, candidate_num=10):
    num_per_cls = out.shape[0]//out.shape[1]
    all_cnt = 0
    def find_next_state(state, board):
        new_states = []
        cur_board = board.copy()
        visited = []
        for i in range(candidate_num):
            cur_num_per_cls = [0, 0, 0, 0, 0]
            for (x, y) in state[0]:
                cur_board[x] = cur_board[x] - cur_board[x]
                cur_num_per_cls[y] += 1
            #     print(x, y, end= '; ')
            # print()
            for (x, y) in visited:
                cur_board[x] = cur_board[x] - cur_board[x]
            #     print(x, y, end= '; ')
            # print()
            # print(cur_num_per_cls, num_per_cls)
            cnt = 0
            while True and cnt < out.shape[0]*out.shape[1]:
                cnt += 1
                # print(cur_board)
                max_per_line = np.max(cur_board, axis=1)
                max_per_line_index = np.argmax(cur_board, axis=1)
                cur_max_index = np.argsort(max_per_line)[-1]
                cur_max = max_per_line[cur_max_index]
                cur_max_cls = max_per_line_index[cur_max_index]
                if cur_num_per_cls[cur_max_cls] < num_per_cls:
                    new_state = [state[0]+ [(cur_max_index, cur_max_cls)], state[1]*cur_board[cur_max_index,cur_max_cls]]
                    # cur_num_per_cls[cur_max_cls] += 1
                    cur_board[cur_max_index] = cur_board[cur_max_index] - cur_board[cur_max_index]
                    visited.append((cur_max_index, cur_max_cls))
                    new_states.append(new_state)
                    # print(cur_board)
                    break
                else:
                    cur_board[cur_max_index, cur_max_cls] = -1
                    # print('conflict', cur_max_index, cur_max_cls)
                
        return new_states

    max_per_line = np.max(out, axis=1)
    max_per_line_index = np.argmax(out, axis=1)
    cur_max_index = np.argsort(max_per_line)[-1]
    cur_max = max_per_line[cur_max_index]
    cur_max_cls = max_per_line_index[cur_max_index]
    # states = [[[(cur_max_index, cur_max_cls)] , out[cur_max_index, cur_max_cls]]]
    states = [[[], 1.0]]
    # print('init' , states)
    states = find_next_state(states[0], out.copy())
    # print('over 1 round', states)
    for i in range(out.shape[0]-1):
        new_states = []
        for i in range(candidate_num):
            board = out.copy()
            for j in range(i):
                for x,y in states[j][0]:
                    board[x, y] = -1
            new_states += find_next_state(states[i], board.copy())
        states = sorted(new_states, key=lambda x: x[1], reverse = True)[:candidate_num]
    # print(states[0], states[1])
    result = np.zeros(out.shape)
    for (x, y) in states[0][0]:
        result[x, y] = 1
    return result

def viterbi_decoding_2(out, candidate_num=3):
    # out: a probability list
    number = len(out)
    way = len(out[0])
    query = number // way
    prob_list = []

    for i in range(number):
        for j in range(way):
            prob_list.append({
                'prob': out[i,j],
                'idx': i,
                'label': j
            })

    # we only keep one copoy of prob list
    # and we keep decode info using other structure
    prob_list = sorted(prob_list, key=lambda x:x['prob'], reverse=True)
    keep_prob_list = [{
        "current_id": 0,
        "trace": {},
        "label_num": [0 for _ in range(way)],
        "prob": 1.0
    }]
    for i in range(number):
        # decode number times
        children = []
        for father in keep_prob_list:
            # bring candidate_num children
            father_id = father['current_id']
            for _ in range(candidate_num):

                # find next suitable ids
                while father_id < number * way:
                    ele = prob_list[father_id]
                    if ele['idx'] not in father['trace'] and father['label_num'][ele['label']] < query:
                        break
                    father_id += 1

                if father_id >= number * way:
                    break
                
                # decode current ele on father id
                ele = prob_list[father_id]
                child = deepcopy(father)
                child['trace'][ele['idx']] = ele['label']
                child['prob'] *= ele['prob']
                child['label_num'][ele['label']] += 1
                child['current_id'] = father_id + 1
                children.append(child)
                # move to next argmax
                father_id += 1
        
        # sort to get top candidate
        children = sorted(children, key=lambda x:x['prob'], reverse=True)
        keep_prob_list = children[:candidate_num]
    
    top1 = keep_prob_list[0]
    return np.eye(5)[[top1['trace'][x] for x in range(number)]]
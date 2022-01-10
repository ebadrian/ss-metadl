from collections import Counter
import os

from src.utils.utils import viterbi_decoding_2

from .base_multi import ProtoMetaLearner, ProtoMultiManager
from .label_propagation import (
    laplacian_label_propagation,
    distance_label_propagation,
    graph_label_propagation,
    map_label_propagation,
    cosine_distance_label_propagation,
    gaussian_label_propagation,
    mct_label_propagation
)
from .embedding_propagation import (
    normalize,
    power_transform,
    embeding_propagation,
    shift_compensation,
    proto_rectification
)
from ...learner.decoder.classifier import Linear, distLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from ...learner.pretrained_encoders.mixup_wrap import resnetmix50, mobilenetmix, resnet152_mix, wide_resnet50_2_mix
import numpy as np
from src.utils import logger, get_root
import time
from src.utils import viterbi_decoding, aggressive_decoding

ONLINE = True

def mean(x):
    return sum(x) / len(x)

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def power_transform_wrapped(supp_x, quer_x, requires_qr=False):
    supp_num = supp_x.shape[0]
    supp_quer_x = torch.cat((supp_x, quer_x), 0)  # (query*way + shot*way) * dim
    supp_quer_x = power_transform(supp_quer_x, requires_qr=requires_qr)
    supp_x = supp_quer_x[:supp_num]
    quer_x = supp_quer_x[supp_num:]
    return supp_x, quer_x

def mixup_data(x, y, lam):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

'''
gconfig = {
    'device': 'cuda:0',
    'pretrain_lr': 0.0001,
    'eval_epoch': 10,
    'eval_tasks': 200,
    'batch_size': 32,
    'lr': 0.0001,
    'epochs': MAX_LOCAL_EPOCH if MAX_LOCAL_EPOCH > 0 else 20000,  # TODO: change to the local max epoch number
    'patience': 20,
    'clip_norm': 1.0,
    'use_pretrain': True,
    'pretrain_epoch': 20000,
    'cls_type': 'linear',
    'train_way': 5,
    'pretrain_shot': 2,
    'first_eval': 0,
    'learnable_distance': False,
    'requires_qr': False,     # require QR reduction in power_transform
    'mixup': True,   # pretrain with manifold 'mixup' (in Phase #2)
    'rotation': True,   # pretrain with 'rotation' loss (in Phase #2)
    'rotation_with_mirror': True,  # add mirror flips for 'rotation' loss
    'alpha': 2.0,    # for mixup pretrain strategy
    'way': 32,    # TODO: keep consistent with config.gin
    'emb_out_layer': 3,   # select the output embedding layer of wrn during inference
    "global_id": 0,
    "emb_prop_valid": 'norm',           # choose from ['pt', 'norm']
    'label_prop_valid': 'euclidean',    # choose from ['euclidean', 'cos', 'graph', 'map', 'lap', 'mct']
    'emb_prop_test': 'norm',
    'label_prop_test': 'map',
    'backbone': 'resnet50'
}
'''
gconfig = {}

def process_data(supp, query, train=True, config=gconfig):
    if train:
        # return [supp, query]
        # load train data
        way, number = len(supp[0]), len(query[0]) // len(supp[0]) + 1
        others = supp[0].size()[1:]
        if config['pretrain_shot'] == 1:
            x = supp[0]
            y = supp[2]
        else:
            x = torch.cat([supp[0], query[0]])
            y = torch.cat([supp[2], query[2]])
            y, slices = y.sort()
            x = x[slices].reshape(way, number, *others)
            y = y.reshape(way, number)
            randidx = torch.randperm(number)[:config['pretrain_shot']]
            x, y = x[:,randidx,:].reshape(way * config['pretrain_shot'], *others), y[:,randidx].reshape(-1)
        
        if config['rotation']:
            # x.shape in CIFAR100: [64=32way*2shot, 3, 28, 28]
            x90 = torch.rot90(x, 1, [2, 3])
            x180 = torch.rot90(x90, 1, [2, 3])
            x270 = torch.rot90(x180, 1, [2, 3])
            if config['rotation_with_mirror']:
                xhm = torch.flip(x, [2])  # horizontal mirror of x
                xvm = torch.flip(x, [3])  # vertical mirror of x
                x_ = torch.cat((x, x90, x180, x270, xhm, xvm), 0)
                y_ = torch.cat((y, y, y, y, y, y), 0)
            else:
                x_ = torch.cat((x, x90, x180, x270), 0)
                y_ = torch.cat((y, y, y, y), 0)
            x = x_
            y = y_
        return [x, y]
    else:
        # load valid data
        return [supp, query]

CLS = {
    'linear': Linear,
    'cosine': distLinear
}

MODEL = {
    'resnet50': resnetmix50,
    'mobilenet': mobilenetmix,
    'wrn50': wide_resnet50_2_mix,
    'resnet152': resnet152_mix
}

def decode_label(sx, qx, emb_prop, label_prop, require_qr, use_hard_mode=None, prob=True, debug=False):
    if emb_prop == 'pt':
        sx, qx = power_transform_wrapped(sx, qx, requires_qr=require_qr)
    else:
        sx = normalize(sx)
        qx = normalize(qx)

    if label_prop == 'map':
        lg = map_label_propagation(qx, sx)
    elif label_prop == 'mct':
        lg = mct_label_propagation(qx, sx)
    elif label_prop == 'graph':
        lg = graph_label_propagation(qx, sx)
    elif label_prop == 'lap':
        lg = laplacian_label_propagation(qx, sx)
    elif label_prop == 'cos':
        lg = cosine_distance_label_propagation(qx, sx)
    else:
        lg = distance_label_propagation(qx, sx)
    if label_prop in ['cos', 'eu', 'euclidean', 'map', 'mct',' graph'] and prob:
        lg = F.softmax(lg, dim=1)
    if label_prop != 'lap':
        lg = lg.detach().cpu().numpy()

    if use_hard_mode == 'agg':
        lg = aggressive_decoding(lg)
    elif use_hard_mode == 'viterbi':
        lg = viterbi_decoding(lg)
    elif use_hard_mode == 'vit':
        lg = viterbi_decoding_2(lg)
    
    return lg

class MyMetaLearner(ProtoMetaLearner):
    def __init__(self, config=gconfig) -> None:
        self.__dict__.update(config)
        self.config = config
        self.emb_layer_best = 0
        self.logger = logger.get_logger('proto_{}'.format(self.global_id))
        
        super().__init__(self.epochs, self.eval_epoch, self.patience, self.eval_tasks, self.batch_size, self.first_eval, self.logger, time_begin=self.time_begin)
        self.device = torch.device(self.device)
        self.logger.info('current hp', config)

    def load_model(self):
        load_model(self.config)

    def create_model(self, class_num):
        self.timer.begin('load pretrained model')
        pre_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../resources/{}.mdl'.format(self.backbone))
        self.model = MODEL[self.backbone](pretrained=True, state_dict_path=pre_path)
        self.model.to(self.device)
        # for origin class training
        times = self.timer.end('load pretrained model')
        self.logger.info('current model', self.model)
        self.logger.info('load time', times, 's')
        self.dim = self.model(torch.randn(2,3,28,28).to(self.device)).size()[-1]
        self.logger.info('detect encoder dimension', self.dim)
        # manually extract the dim
        self.cls = CLS[self.cls_type](self.dim, class_num).to(self.device)
        self.rotate_cls = nn.Sequential(nn.Linear(self.dim, 6)).to(self.device)
        self.rotate_label = torch.tensor(np.arange(6)).unsqueeze(dim=1).repeat(1, self.way * self.pretrain_shot).flatten().to(self.device)
        self.opt_pretrain = optim.Adam(list(self.model.parameters()) + list(self.cls.parameters()) + 
                                        list(self.rotate_cls.parameters()), lr=self.pretrain_lr)
        
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.best_model = None
    
    def on_train_begin(self, epoch):
        self.model.train()
        self.cls.train()
        self.rotate_cls.train()
        self.err_list = []
        self.acc_list = []
        self.opt.zero_grad()
        self.opt_pretrain.zero_grad()
        self.mode = 'pretrain'
        return True

    def on_train_end(self, epoch):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= len(self.err_list)
        nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
        self.opt.step()
        err = mean(self.err_list)
        acc = mean(self.acc_list)
        self.logger.info('epoch %2d mode: %s error %.6f acc %.6f' % (epoch, self.mode, err, acc))
        self.logger.info('extract data', self.timer.query_time_by_name('extract data', lambda x:sum(x[-self.batch_size:])))
        self.logger.info('to device', self.timer.query_time_by_name('to device', lambda x:sum(x[-self.batch_size:])))
        self.logger.info('do cal', self.timer.query_time_by_name('do cal', lambda x:sum(x[-self.batch_size:])), '\n')
    
    def mini_epoch(self, train_pipe, epoch, iters):
        # use pretrain
        # we need to select part of examples for training
        self.timer.begin('extract data')
        x, y = train_pipe.recv()
        train_pipe.send(True)
        self.timer.end('extract data')
        self.timer.begin('to device')
        x = x.to(self.device)
        y = y.to(self.device)
        self.timer.end('to device')
        self.timer.begin('do cal')
        lam = np.random.beta(self.alpha, self.alpha)  # draw from the Beta distribution, default alpha = 2.0 (then 0 < lam < 1)
        input_var, target_var = Variable(x), Variable(y)
        feature, target_a, target_b = self.model(input_var, target_var, mixup_hidden=True, lam=lam)
        logit = self.cls(feature)
        loss = mixup_criterion(F.cross_entropy, logit, target_a, target_b, lam)
        rotate_logit = self.rotate_cls(feature)
        rloss = F.cross_entropy(rotate_logit, self.rotate_label)
        loss = 0.5 * loss + 0.5 * rloss
        loss.backward()
        self.err_list.append(loss.item())
        _, predicted = torch.max(logit.data, 1)
        acc = (lam * predicted.eq(target_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
        acc /= y.size(0)
        self.acc_list.append(acc)
        self.timer.end('do cal')
    
    def eval_one_episode(self, valid_pipe):
        with torch.no_grad():
            t1 = time.time()
            supp, query = valid_pipe.recv()
            valid_pipe.send(True)
            t2 = time.time()
            _, slices = supp[1].to(self.device).sort()
            supp_x = self.model( supp[0].to(self.device)[slices], emb_out_layer=self.emb_out_layer)
            quer_x = self.model(query[0].to(self.device), emb_out_layer=self.emb_out_layer)
            t3 = time.time()
            acc_list = []
            for i in range(len(supp_x)):
                lg = decode_label(supp_x[i], quer_x[i], self.emb_prop_valid, self.label_prop_valid, self.requires_qr, self.use_hard_mode)
                acc = (lg.argmax(1) == query[1].detach().cpu().numpy()).mean()
                acc_list.append(acc)
            self.acc_idx_max.append(np.argmax(acc_list))
            return max(acc_list)

    def on_eval_begin(self, epoch):
        self.model.eval()
        self.time_dict = {}
        self.acc_idx_max = []
        return True
    
    def on_eval_end(self, epoch, acc, patience_now):
        counter = Counter(self.acc_idx_max)
        self.logger.info('max idx distribution', counter)
        self.emb_layer_best = counter.most_common(1)[0][0]
        return True

    def save_model(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(self.global_id))
        os.makedirs(path, exist_ok=True)
        if hasattr(self, 'emb_layer_best'):
            emb_layer = self.emb_layer_best
        elif self.emb_out_layer == -2:
            emb_layer = 1
        else:
            emb_layer = self.emb_out_layer
        torch.save({
            'model': self.model,
            'emb_layer_best': emb_layer
        }, os.path.join(path, 'tmp.pt'))
    
    def make_learner(self):
        return MyMultiManager(self.model, dis_scaler=None if not self.learnable_distance else self.dis_scaler, emb_out_layer=self.emb_out_layer if self.emb_out_layer != -2 else self.emb_layer_best)

class MyMultiManager(ProtoMultiManager):
    def __init__(self, model=None, ldis=None, out_layer=-1, config=gconfig) -> None:
        self.model = model
        self.ldis = ldis
        self.out_layer = out_layer
        self.config = config
        self.loaded = False
        # self.logger = logger.get_logger('proto_' + str(self.config['global_id']))

    def load_model(self, path, device='auto'):
        if self.loaded:
            return
        other = torch.load(os.path.join(path, 'config.pt'))
        self.config = other['config']
        self.out_layer = other['out_layer']
        self.loaded = True
        if device == 'auto':
            device = self.config['device']
        self.config['device'] = device
        self.model = torch.load(os.path.join(path, 'model.pt'), map_location=device)
        if os.path.exists(os.path.join(path, 'ldis.pt')):
            self.ldis = torch.load(os.path.join(path, 'ldis.pt'), map_location=device)

    def save_model(self, path):
        torch.save(self.model, os.path.join(path, 'model.pt'))
        torch.save({
            'config': self.config,
            'out_layer': self.out_layer
        }, os.path.join(path, 'config.pt'))
    
    def to(self, device):
        self.model.to(device)

    def eval_one_episode(self, supp_x, supp_y, img, device):
        self.model.to(device)
        supp_x = supp_x.to(device)
        supp_y = supp_y.to(device)
        img = img.to(device)
        self.model.eval()

        # print('global id', self.config['global_id'], 'get inside one episode')
        max_test_batch = 5000
        with torch.no_grad():
            _, slices = supp_y.sort()
            supp_x = self.model(supp_x[slices], emb_out_layer=self.out_layer)
            begin_idx = 0
            quer_xs = []
            while begin_idx < img.shape[0]:
                quer_x = self.model(img[begin_idx:begin_idx + max_test_batch], emb_out_layer=self.out_layer)
                quer_xs.append(quer_x)
                begin_idx += max_test_batch
            if img.shape[0] == 0:
                return []
            quer_x = torch.cat(quer_xs)
            lg = decode_label(supp_x, quer_x, self.config['emb_prop_test'], self.config['label_prop_test'], self.config['requires_qr'], use_hard_mode=self.config['use_hard_mode'], prob=True, debug=True)
            return lg

def load_model(config=gconfig):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'proto_{}'.format(config['global_id']))
    path_to_tmp = os.path.join(path, 'tmp.pt')
    device = torch.device(config['device'])
    m = torch.load(path_to_tmp, map_location=device)
    model = m['model'].to(device)
    emb_layer_best = m['emb_layer_best']
    return MyMultiManager(model, None, emb_layer_best, config)
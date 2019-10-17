import os

import torch
import numpy as np


def build_model_name(epochs, id_exp):
    file_seed = id_exp if id_exp is not None else np.random.randint(int(10 ** 5))
    name = str(epochs) + 'epochs-' + str(file_seed)
    return name


def common_processing(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        args.device = torch.device("cuda:" + str(args.gpu))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    else:
        args.device = "cpu"
    print('Device used:', args.device)

    args.model_name = build_model_name(args.epochs, args.id)

    args.paths = {'save': './models/{}/{}.pt'.format(args.dataset, args.model_name),
                  'load': './models/{}/{}-cpu.pt'.format(args.dataset, args.model_name),
                  'results': './results/{}/{}.txt'.format(args.dataset, args.model_name),
                  'train_results': './results/{}/train{}.txt'.format(args.dataset, args.model_name)}
    return args


class EarlyStopping(object):
    """ Based on gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    """
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta



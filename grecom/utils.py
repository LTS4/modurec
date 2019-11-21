import os

import torch
import numpy as np
import h5py
import scipy.sparse as sp


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

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

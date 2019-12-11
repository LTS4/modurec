import os
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import h5py
import numpy as np
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch import optim, tensor
from tqdm import tqdm
import pandas as pd

from grecom.data import split_data


def _load_model_data(args, rating_type):
    data = {}
    f = h5py.File(os.path.join(args.data_path, "data.h5"), "r")
    data['u_counts'] = f['ca_data']['user_counts'][:]
    data['v_counts'] = f['ca_data']['item_counts'][:]
    if rating_type == 'U':
        data['input_size'] = len(data['v_counts'])
        data['epoch_size'] = len(data['u_counts'])
        data['row'] = f['cf_data']['row'][:]
        data['col'] = f['cf_data']['col'][:]
    else:
        data['input_size'] = len(data['u_counts'])
        data['epoch_size'] = len(data['v_counts'])
        data['row'] = f['cf_data']['col'][:]
        data['col'] = f['cf_data']['row'][:]
    data['rating'] = f['cf_data']['rating'][:]
    if 'time' in f['cf_data']:
        data['time'] = {}
        for key, value in f['cf_data']['time'].items():
            data['time'][key] = value[:]
    with h5py.File(os.path.join(
            args.split_path, str(args.split_id), "train_mask.h5"), "r") as f:
        data['train_mask'] = f["train_mask"][:]
    return data


class DataGenerator():
    def __init__(self, data, args, use_time=False, max_mem=1e9):
        self.use_time = use_time
        self.args = args
        self.batch_size = max_mem // data['input_size']
        self.data = data
        self.data_dense = None
        if self.batch_size > data['epoch_size']:
            self.data_dense = self.load_all_data()

    def load_all_data(self):
        dd_index = (self.data['row'], self.data['col'])
        dd_shape = (self.data['epoch_size'], self.data['input_size'])
        dd = {}
        dd['x'] = coo_matrix(
            (self.data['rating'], dd_index),
            shape=dd_shape, dtype=np.float32).toarray()
        if not self.use_time:
            return dd
        x_time = np.stack([
            coo_matrix(
                (value, dd_index), shape=dd_shape, dtype=np.float32).toarray()
            for value in self.data['time'].values()
        ])
        dd['time'] = np.transpose(x_time, (1, 2, 0))
        dd_mask = tuple(x[self.data['train_mask'] == 1] for x in dd_index)
        dd['train_mask'] = coo_matrix(
            (np.ones_like(dd_mask[0]), dd_mask),
            shape=dd_shape, dtype=np.float32).toarray()
        for key, value in dd.items():
            dd[key] = tensor(value).to(self.args.device)
        return dd

    def next(self):
        if self.data_dense is not None:
            return self.data_dense


def _get_training_pred(args, dd, model):
    x_tr = dd['x'] * dd['train_mask']
    pred = model(x_tr)
    if model.requires_time:
        x_time = dd['time'] * dd['train_mask'].unsqueeze(-1)
        pred = model(x_tr, x_time)
    return pred


def train_val_batch(args, dd, model, epoch, reg):
    model.train()
    p = _get_training_pred(args, dd, model)
    x_tr = dd['x'] * dd['train_mask']
    loss = (
        F.mse_loss(x_tr[x_tr != 0], p[x_tr != 0])
        + model.get_reg_loss())
    loss.backward()

    model.eval()
    if reg is None:
        reg = pd.DataFrame({'epoch': [], 'tr_rmse': [], 'te_rmse': []})
    p = _get_training_pred(args, dd, model)
    x_te = dd['x'] * (1 - dd['train_mask'])
    reg = reg.append({
        'epoch': epoch,
        'tr_rmse': F.mse_loss(x_tr[x_tr != 0], p[x_tr != 0]).item() ** (1/2),
        'te_rmse': F.mse_loss(x_te[x_te != 0], p[x_te != 0]).item() ** (1/2)
    }, ignore_index=True)
    return reg


def train(args):
    rating_type, model_str = args.model.split('-')
    data = _load_model_data(args, rating_type)
    model_module = import_module(f"grecom.model")
    model = getattr(model_module, model_str)(args, data['input_size'])
    data_gen = DataGenerator(data, args, model.requires_time)
    dd = data_gen.next()
    reg = None
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.96)
    ex = ThreadPoolExecutor()
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        th_model = ex.submit(train_val_batch, *(args, dd, model, epoch, reg))
        th_data = ex.submit(data_gen.next)
        reg = th_model.result()
        dd = th_data.result()
        optimizer.step()
        scheduler.step()
    print(reg)


def run_experiment(args):
    """Run experiment on testing scenario. Performs the following steps:
    - Split data according to scenario
    - Train model
    - Evaluate model and save results

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    split_data(args)
    train(args)

from importlib import import_module
import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from torch import tensor


def prepare_data(args):
    """Preprocess data if necessary

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    folder = os.path.join(args.data_path, args.dataset)
    args.data_path = folder
    if not args.preprocess:
        if os.path.exists(folder):
            return args
    data_module = import_module(f"grecom.data.{args.dataset}")
    getattr(data_module, 'preprocess')(args)
    return args


def split_data(args):
    if args.split_type == 'predefined':
        data_module = import_module(f"grecom.data.{args.dataset}")
        getattr(data_module, 'split_predefined')(args)
    elif args.split_type == 'random':
        save_path = os.path.join(args.split_path, str(args.split_id))
        if os.path.exists(save_path):
            return
        with h5py.File(os.path.join(args.data_path, "data.h5"), "r") as f:
            row = f['cf_data']['row'][:]
        _, test_inds = train_test_split(
            np.arange(len(row)), train_size=args.train_prop,
            random_state=args.split_id)
        train_mask = np.ones(len(row))
        train_mask[test_inds] = 0
        os.makedirs(save_path)
        with h5py.File(os.path.join(save_path, "train_mask.h5"), "w") as f:
            f["train_mask"] = train_mask


class DataGenerator():
    def __init__(self, args, model_class, rating_type, max_mem=1e9):
        self.args = args
        self.use_time = model_class.requires_time
        self.use_fts = model_class.requires_fts
        self.rating_type = rating_type
        self.data = self.load_data_from_h5()
        self.batch_size = max_mem // self.data['input_size']
        self.data_dense = None
        if self.batch_size > self.data['epoch_size']:
            self.data_dense = self.create_dense_data()

    def load_data_from_h5(self):
        data = {}
        f = h5py.File(os.path.join(self.args.data_path, "data.h5"), "r")
        if self.rating_type == 'U':
            data['input_size'] = len(data['v_counts'])
            data['epoch_size'] = len(data['u_counts'])
            data['row'] = f['cf_data']['row'][:]
            data['col'] = f['cf_data']['col'][:]
            if self.use_fts:
                data['fts'] = (
                    f['ca_data']['user_fts'][:],
                    f['ca_data']['item_fts'][:]
                )
                data['counts'] = (
                    f['ca_data']['user_counts'][:],
                    f['ca_data']['item_counts'][:]
                )
        else:
            data['input_size'] = len(data['u_counts'])
            data['epoch_size'] = len(data['v_counts'])
            data['row'] = f['cf_data']['col'][:]
            data['col'] = f['cf_data']['row'][:]
            if self.use_fts:
                data['fts'] = (
                    f['ca_data']['item_fts'][:],
                    f['ca_data']['user_fts'][:]
                )
                data['counts'] = (
                    f['ca_data']['item_counts'][:],
                    f['ca_data']['user_counts'][:]
                )
        data['rating'] = f['cf_data']['rating'][:]
        if self.use_time:
            data['time'] = {}
            for key, value in f['cf_data']['time'].items():
                data['time'][key] = value[:]
        with h5py.File(os.path.join(
                self.args.split_path, str(self.args.split_id), "train_mask.h5"
        ), "r") as f:
            data['train_mask'] = f["train_mask"][:]
        return data

    def create_dense_data(self):
        dd_index = (self.data['row'], self.data['col'])
        dd_shape = (self.data['epoch_size'], self.data['input_size'])
        dd = {}
        dd['x'] = coo_matrix(
            (self.data['rating'], dd_index),
            shape=dd_shape, dtype=np.float32).toarray()
        if self.use_time:
            x_time = np.stack([
                coo_matrix(
                    (value, dd_index), shape=dd_shape,
                    dtype=np.float32).toarray()
                for value in self.data['time'].values()
            ])
            dd['time'] = np.transpose(x_time, (1, 2, 0))
        if self.use_fts:
            dd['fts'] = self.data['fts']
            dd['counts'] = self.data['counts']
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

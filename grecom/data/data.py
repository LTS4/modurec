from importlib import import_module
import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import torch
from torch import tensor


def prepare_data(args):
    """Preprocess data if necessary.

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
    """Split data into train and test as defined by the execution parameters.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
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
    """Loads batches with the data the model requires. If the data is small,
    it is loaded entirely on RAM.
    """

    def __init__(self, args, model_class, rating_type, batch_size=None,
                 max_mem=2e9):
        """Default initialization.

        :param args: Dictionary with execution arguments
        :type args: Namespace
        :param model_class: Model class with static attributes that define the
        data required.
        :type model_class: torch.nn.Module
        :param rating_type: 'I' for item ratings, 'U' for user ratings.
        :type rating_type: str
        :param max_mem: RAM/VRAM maximum size. In practice, this only accounts
        for one matrix of collaborative data. Defaults to 2 GB.
        :type max_mem: int, optional
        """
        self.args = args
        self.use_time = model_class.requires_time
        self.use_fts = model_class.requires_fts
        self.use_graph = model_class.requires_graph
        self.rating_type = rating_type
        self.data = self.load_data_from_h5()
        self.input_size = self.data['input_size']
        self.epoch_size = self.data['epoch_size']
        if batch_size is None:
            self.batch_size = (max_mem // (4 * self.input_size))  # float: 4
        self.data_dense = None
        if self.batch_size > self.epoch_size:
            self.data_dense = self.create_dense_data()
        else:
            self.data_csr = self.create_csr_data()
            self.i = self.batch_size
            self.prev_i = 0

    def load_data_from_h5(self):
        """Loads the h5py data (collaborative data in COO format). Takes into
        account the type of input (user or item ratings).

        :return: Dictionary with the data loaded.
        :rtype: dict
        """
        data = {}
        f = h5py.File(os.path.join(self.args.data_path, "data.h5"), "r")
        if self.rating_type == 'U':
            indices = (f['cf_data']['row'][:], f['cf_data']['col'][:])
            p = ('user', 'item')  # prefixes
        else:
            indices = (f['cf_data']['col'][:], f['cf_data']['row'][:])
            p = ('item', 'user')
        data['input_size'] = len(f['ca_data'][f'{p[1]}_counts'])
        data['epoch_size'] = len(f['ca_data'][f'{p[0]}_counts'])
        shape = (data['epoch_size'], data['input_size'])
        data['x'] = coo_matrix(
            (f['cf_data']['rating'][:], indices),
            shape=shape, dtype=np.float32)
        if self.use_fts:
            data['fts'] = (
                f['ca_data'][f'{p[0]}_fts'][:],
                f['ca_data'][f'{p[1]}_fts'][:]
            )
        if self.use_graph:
            data['graph'] = (
                {
                    'edge_index': f['ca_data'][f'{p[0]}_graph_idx'][:],
                    'edge_weight': f['ca_data'][f'{p[0]}_graph_ws'][:]
                },
                {
                    'edge_index': f['ca_data'][f'{p[1]}_graph_idx'][:],
                    'edge_weight': f['ca_data'][f'{p[1]}_graph_ws'][:]
                }
            )
        if self.use_time:
            data['time'] = {}
            for key, value in f['cf_data']['time'].items():
                data['time'][key] = coo_matrix(
                    (value[:], indices), shape=shape, dtype=np.float32)
        with h5py.File(os.path.join(
                self.args.split_path, str(self.args.split_id), "train_mask.h5"
        ), "r") as f:
            mask = tuple(x[f["train_mask"][:] == 1] for x in indices)
            data['train_mask'] = coo_matrix(
                (np.ones_like(mask[0]), mask),
                shape=shape, dtype=np.float32)
        return data

    def create_dense_data(self):
        """Transforms the COO matrices into dense matrices.

        :return: Dictionary with the data as dense matrices.
        :rtype: dict
        """
        data = {}
        data['x'] = self.data['x'].toarray()
        if self.use_graph:
            data['graph'] = self.data['graph']
        if self.use_time:
            x_time = np.stack([
                value.toarray() for value in self.data['time'].values()
            ])
            data['time'] = np.transpose(x_time, (1, 2, 0))
        data['train_mask'] = self.data['train_mask'].toarray()
        for key, value in data.items():
            # For graph key, we have a tuple of dicts with array values
            if key == 'graph':
                temp_list = []
                for g in value:
                    g['edge_index'] = tensor(
                        g['edge_index'], dtype=torch.long
                    ).to(self.args.device)
                    g['edge_weight'] = tensor(
                        g['edge_weight'], dtype=torch.float
                    ).to(self.args.device)
                    temp_list.append(g)
                data[key] = tuple(temp_list)
            elif isinstance(value, tuple):
                data[key] = tuple(
                    tensor(x, dtype=torch.float).to(self.args.device)
                    for x in value)
            else:
                data[key] = tensor(
                    value, dtype=torch.float).to(self.args.device)
        return data

    def create_csr_data(self):
        """Transforms the COO matrices into CSR matrices.

        :return: Dictionary with the data as CSR matrices.
        :rtype: dict
        """
        data = {}
        for key, value in self.data.items():
            if isinstance(value, coo_matrix):
                data[key] = value.tocsr()
            else:
                data[key] = value
        return data

    def __iter__(self):
        if self.data_dense is None:
            self.i = self.batch_size
            self.prev_i = 0
        return self

    def __next__(self):
        """Returns next batch.

        :return: Dictionary with the data
        :rtype: dict
        """
        if self.data_dense is not None:
            return self.data_dense
        row_indices = self.data_csr.x.indptr[self.prev_i:i]
        prev_j = row_indices[0]
        for j in row_indices[1:]:
            indices = self.data_csr.x.indices[prev_j:j]
            values = self.data_csr.x.data[prev_j:j]
        return None

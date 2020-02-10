from importlib import import_module
import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
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

    def __init__(self, args, model_class, rating_type, max_mem=11e9):
        """Default initialization.

        :param args: Dictionary with execution arguments
        :type args: Namespace
        :param model_class: Model class with static attributes that define the
        data required.
        :type model_class: torch.nn.Module
        :param rating_type: 'I' for item ratings, 'U' for user ratings.
        :type rating_type: str
        :param max_mem: RAM/VRAM maximum size. Defaults to 11 GB.
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
        self.new_epoch = False
        self.batch_size = args.batch_size
        self.data_dense = None
        self.patch_inds = np.arange(self.epoch_size)
        if self.batch_size == 0:
            self.data_dense = self.create_dense_data()
        else:
            if self.use_graph:
                raise NotImplementedError(
                    "Sparse graph convolutions not implemented")
            self.data_csr = self.create_csr_data()
            #self.data_fast = self.create_h5dense()
            np.random.shuffle(self.patch_inds)
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
            data['ft_x'] = (
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
        ), "r") as fm:
            mask = tuple(x[fm["train_mask"][:] == 1] for x in indices)
            data['train_mask'] = coo_matrix(
                (np.ones_like(mask[0]), mask),
                shape=shape, dtype=np.float32)
            mv = f['cf_data']['rating'][fm["train_mask"][:] == 1].mean()
        # Count number of ratings (if zero in train, impose/content in val)
        v1 = np.zeros_like(f['ca_data'][f'{p[0]}_counts'][:])
        v2 = np.zeros_like(f['ca_data'][f'{p[1]}_counts'][:])
        ind, cts = np.unique(mask[0], return_counts=True)
        v1[ind] = cts
        ind, cts = np.unique(mask[1], return_counts=True)
        v2[ind] = cts
        data['ft_n'] = (v1, v2, mv)
        data['rating_range'] = (data['x'].data.min(), data['x'].data.max())
        return data

    def create_h5dense(self):
        h5_densepath = os.path.join(self.args.data_path, "data_dense.h5")
        save_flag = True
        if os.path.isfile(h5_densepath):
            save_flag = False
        f = h5py.File(h5_densepath, "a")
        r = h5py.File(h5_densepath, "r")
        data = {}
        for key, value in self.data.items():
            if isinstance(value, coo_matrix):
                if save_flag:
                    f[key] = value.toarray()
                data[key] = r[key]
            elif key == 'time':
                if save_flag:
                    f.create_dataset(
                        key, (self.epoch_size, self.input_size, len(value)))
                    for i, t_value in enumerate(value.values()):
                        f[key][..., i] = t_value.toarray()
                data[key] = r[key]
            else:
                data[key] = value
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
            elif key == 'time':
                data['time'] = {}
                for t_key, t_value in value.items():
                    data['time'][t_key] = t_value.tocsr()
            else:
                data[key] = value
        return data

    def create_dense_data(self):
        """Transforms the COO matrices into dense matrices.
        TODO: simplify, new dict not needed

        :return: Dictionary with the data as dense matrices.
        :rtype: dict
        """
        data = {}
        data['patch_inds'] = tensor(self.patch_inds).to(self.args.device)
        data['x'] = self.data['x'].toarray()
        data['ft_n'] = self.data['ft_n']
        if self.use_graph:
            data['graph'] = self.data['graph']
        if self.use_time:
            x_time = np.stack([
                value.toarray() for value in self.data['time'].values()
            ])
            data['time'] = np.transpose(x_time, (1, 2, 0))
        if self.use_fts:
            data['ft_x'] = self.data['ft_x']
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

    def create_patch_data(self):
        sp_data = self.data_csr
        patch_inds = self.patch_inds[self.prev_i:self.i]
        data = {}
        data['patch_inds'] = tensor(patch_inds).to(self.args.device)
        data['input_size'] = tensor(
            sp_data['input_size'], dtype=torch.float).to(self.args.device)
        data['epoch_size'] = tensor(
            sp_data['epoch_size'], dtype=torch.float).to(self.args.device)
        shape = (len(patch_inds), sp_data['input_size'])
        indptr = sp_data['x'][patch_inds].indptr
        indices = sp_data['x'][patch_inds].indices
        tuple_data = (sp_data['x'][patch_inds].data, indices, indptr)
        data['x'] = tensor(
            csr_matrix(tuple_data, shape=shape).toarray(),
            dtype=torch.float).to(self.args.device)
        data['ft_n'] = (
            tensor(
                sp_data['ft_n'][0][patch_inds], dtype=torch.float
            ).to(self.args.device),
            tensor(sp_data['ft_n'][1], dtype=torch.float).to(self.args.device),
            sp_data['ft_n'][2]
        )
        if self.use_fts:
            data['ft_x'] = (
                tensor(
                    sp_data['ft_x'][0][patch_inds], dtype=torch.float
                ).to(self.args.device),
                tensor(
                    sp_data['ft_x'][1], dtype=torch.float).to(self.args.device)
            )
        if self.use_time:
            # The indices can be reused because: x_ij = 0 <-> time_ij = 0
            data_time = (
                value[patch_inds].data for value in sp_data['time'].values()
            )
            x_time = np.stack([
                csr_matrix((data, indices, indptr), shape=shape).toarray()
                for data in data_time
            ])
            data['time'] = tensor(
                np.transpose(x_time, (1, 2, 0)), dtype=torch.float
            ).to(self.args.device)
        # Train mask is sparser, so we cannot use the previous indices
        mask = sp_data['train_mask']
        indptr = mask[patch_inds].indptr
        indices = mask[patch_inds].indices
        tuple_data = (mask[patch_inds].data, indices, indptr)
        data['train_mask'] = tensor(
            csr_matrix(tuple_data, shape=shape).toarray(),
            dtype=torch.float).to(self.args.device)
        return data

    def _subset(self, array, patch_inds):
        if self.rating_type == 'U':
            return np.swapaxes(array, 0, 1)[patch_inds,...]
        else:
            return array[patch_inds,...]

    def create_patch_data_fast(self):
        ddata = self.data_fast
        patch_inds = np.sort(self.patch_inds[self.prev_i:self.i])
        data = {}
        data['patch_inds'] = tensor(patch_inds).to(self.args.device)
        data['input_size'] = tensor(
            ddata['input_size'], dtype=torch.float).to(self.args.device)
        data['epoch_size'] = tensor(
            ddata['epoch_size'], dtype=torch.float).to(self.args.device)
        data['x'] = tensor(
            self._subset(ddata['x'], patch_inds),
            dtype=torch.float).to(self.args.device)
        data['ft_n'] = (
            tensor(
                ddata['ft_n'][0][patch_inds], dtype=torch.float
            ).to(self.args.device),
            tensor(ddata['ft_n'][1], dtype=torch.float).to(self.args.device),
            ddata['ft_n'][2]
        )
        if self.use_fts:
            data['ft_x'] = (
                tensor(
                    ddata['ft_x'][0][patch_inds], dtype=torch.float
                ).to(self.args.device),
                tensor(
                    ddata['ft_x'][1], dtype=torch.float).to(self.args.device)
            )
        if self.use_time:
            data['time'] = tensor(
                self._subset(ddata['time'], patch_inds),
                dtype=torch.float).to(self.args.device)
        data['train_mask'] = tensor(
            self._subset(ddata['train_mask'], patch_inds),
            dtype=torch.float).to(self.args.device)
        return data

    def __iter__(self):
        if self.data_dense is None:
            np.random.shuffle(self.patch_inds)
            self.i = self.batch_size
            self.prev_i = 0
            self.new_epoch = True
        return self

    def __next__(self):
        """Returns next batch. Autoresets when finished.

        :return: Dictionary with the data
        :rtype: dict
        """
        self.new_epoch = False
        if self.data_dense is not None:
            self.new_epoch = True
            return self.data_dense
        # Resets and gives first batch of next epoch
        if self.prev_i == self.epoch_size:
            np.random.shuffle(self.patch_inds)
            self.i = self.batch_size
            self.prev_i = 0
            self.new_epoch = True
        data = self.create_patch_data()
        #data = self.create_patch_data_fast()
        # Update batch indices
        self.prev_i = self.i
        self.i = min(self.i + self.batch_size, self.epoch_size)
        return data

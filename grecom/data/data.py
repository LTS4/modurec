from importlib import import_module
import os
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


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
        with h5py.File(os.path.join(args.data_path, "data.h5"), "r") as f:
            row = f['cf_data']['row'][:]
        _, test_inds = train_test_split(
            np.arange(len(row)), train_size=args.train_prop,
            random_state=args.split_id)
        train_mask = np.ones(len(row))
        train_mask[test_inds] = 0
        save_path = os.path.join(args.split_path, str(args.split_id))
        os.makedirs(save_path)
        with h5py.File(os.path.join(save_path, "train_mask.h5"), "w") as f:
            f["train_mask"] = train_mask

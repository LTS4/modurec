import os
import pickle

import h5py
import numpy as np
import pandas as pd
from torch_geometric.data import download_url, extract_zip

from grecom.data.utils import get_reltime, load_matlab_file
from grecom.data.geometric import create_similarity_graph
import scipy.sparse as sp
import stanfordnlp
import re
import string

DATASET = 'yahoo_music'
URL = 'https://github.com/riannevdberg/gc-mc/raw/master/gcmc/data/yahoo_music/training_test_dataset.mat'


def initialize_data(data_path):
    """Creates file that will contained the preprocessed data

    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    """
    os.makedirs(data_path, exist_ok=True)
    with h5py.File(os.path.join(data_path, "data.h5"), "w") as f:
        f.create_group("cf_data")  # cf = collaborative filtering
        f.create_group("ca_data")  # ca = content analysis
        f['cf_data'].create_group('time')


def download(raw_path):
    """Downloads the raw data from URL

    :param raw_path: download path
    :type raw_path: str
    """
    path = download_url(URL, os.path.join(raw_path, DATASET))


def _read_ratings(raw_path):
    """Read rating data

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :return: Dataframe with user/item ids, rating and timestamp
    :rtype: pandas.DataFrame
    """
    return load_matlab_file(
        os.path.join(raw_path, 'training_test_dataset.mat'), 'M')


def _process_id_mappings(ratings, data_path):
    """Reindexes the user and item ids and sorts the table.
    Saves the mappings and the counts.

    :param ratings: Dataframe with user/item ids
    :type ratings: pandas.DataFrame
    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    :return: Dataframe with remapped user/item ids
    :rtype: pandas.DataFrame
    """
    u_nodes = np.where(ratings)[0].astype(np.int64)
    v_nodes = np.where(ratings)[1].astype(np.int64)
    u_ids, u_counts = np.unique(u_nodes, return_counts=True)
    v_ids, v_counts = np.unique(v_nodes, return_counts=True)
    # ar = absolute -> relative
    dict_user_ar = {id_: i for i, id_ in enumerate(u_ids)}
    dict_item_ar = {id_: i for i, id_ in enumerate(v_ids)}
    ratings['user_id'] = ratings['user_id'].map(dict_user_ar)
    ratings['item_id'] = ratings['item_id'].map(dict_item_ar)
    ratings = ratings.sort_values(['user_id', 'item_id'])
    with open(os.path.join(data_path, "dict_user_ar.pickle"), "wb") as f:
        pickle.dump(dict_user_ar, f)
    with open(os.path.join(data_path, "dict_item_ar.pickle"), "wb") as f:
        pickle.dump(dict_item_ar, f)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']['user_counts'] = u_counts
        f['ca_data']['item_counts'] = v_counts
    return ratings


def _create_rating_matrix(ratings, data_path):
    """Saves the rating matrix in COO format

    :param ratings: Dataframe with user/item ids
    :type ratings: pandas.DataFrame
    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    """
    u_nodes = np.where(ratings)[0].astype(np.int64)
    v_nodes = np.where(ratings)[1].astype(np.int64)
    rating = ratings[np.where(ratings)]
    u_unique, u_counts = np.unique(u_nodes, return_counts=True)
    v_unique, v_counts = np.unique(v_nodes, return_counts=True)
    print(u_unique[:100])
    print(ratings.sum(1)[:100])
    assert len(u_unique) == (u_unique[-1] + 1)
    assert len(v_unique) == (v_unique[-1] + 1)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['cf_data']["row"] = row
        f['cf_data']['col'] = col
        f['cf_data']['rating'] = rating
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']['user_counts'] = u_counts
        f['ca_data']['item_counts'] = v_counts


def process_rating_data(raw_path, data_path):
    """Creates id mappings, rating and time matrices

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    """
    ratings = _read_ratings(raw_path)
    _create_rating_matrix(ratings, data_path)


def preprocess_user_features(raw_path, data_path):
    """Read and preprocess user features. It uses the age, gender
    and occupation.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    num_users = load_matlab_file(raw_path, 'M').shape[0]
    user_fts = np.eye(num_users)
    indices, weights = create_similarity_graph(user_fts)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']['user_fts'] = user_fts
        f['ca_data']['user_graph_idx'] = indices
        f['ca_data']['user_graph_ws'] = weights


def preprocess_item_features(raw_path, data_path):
    """Read and preprocess item features. It uses the year and genre.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    item_fts = load_matlab_file(path_dataset, 'W_tracks')
    indices, weights = create_similarity_graph(item_fts)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']["item_fts"] = item_fts
        f['ca_data']['item_graph_idx'] = indices
        f['ca_data']['item_graph_ws'] = weights


def preprocess(args):
    """Preprocess and save the data

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    raw_path = os.path.join(args.raw_path, DATASET)
    data_path = args.data_path
    if not os.path.exists(raw_path):
        download(args.raw_path)
    initialize_data(data_path)
    process_rating_data(raw_path, data_path)
    preprocess_user_features(raw_path, data_path)
    preprocess_item_features(raw_path, data_path)


def split_predefined(args):
    """Loads test set, maps ids and creates a training mask.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    save_path = os.path.join(args.split_path, str(args.split_id))
    if os.path.exists(save_path):
        return
    raw_path = os.path.join(args.raw_path, DATASET)
    Otest = load_matlab_file(raw_path, 'Otest')
    test_inds = np.array(
        [[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    with h5py.File(os.path.join(args.data_path, "data.h5"), "r") as f:
        row = f['cf_data']['row'][:]
        col = f['cf_data']['col'][:]
    train_mask = np.ones(len(row), dtype=int)
    j = 0
    for i in range(len(row)):
        if (test_inds[j] == (row[i], col[i])).all():
            j += 1
            train_mask[i] = 0
            if j == len(test_inds):
                break
    os.makedirs(save_path)
    with h5py.File(os.path.join(save_path, "train_mask.h5"), "w") as f:
        f["train_mask"] = train_mask

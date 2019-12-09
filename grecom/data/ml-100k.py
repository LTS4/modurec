import os
import pickle

import h5py
import numpy as np
import pandas as pd
from torch_geometric.data import download_url, extract_zip

from grecom.data.utils import get_reltime

DATASET = 'ml-100k'
URL = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'


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
    path = download_url(URL, raw_path)
    extract_zip(path, raw_path)
    os.unlink(path)


def _read_ratings(raw_path):
    """Read rating data

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :return: Dataframe with user/item ids, rating and timestamp
    :rtype: pandas.DataFrame
    """
    colnames = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        os.path.join(raw_path, 'u.data'),
        sep='\t', header=None, names=colnames)
    return ratings


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
    u_ids, u_counts = np.unique(ratings.user_id.values, return_counts=True)
    v_ids, v_counts = np.unique(ratings.item_id.values, return_counts=True)
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
    row = ratings.user_id.values
    col = ratings.item_id.values
    rating = ratings.rating.values
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['cf_data']["row"] = row
        f['cf_data']['col'] = col
        f['cf_data']['rating'] = rating


def _create_time_matrices(ratings, data_path):
    """Saves the time matrices in COO format

    :param ratings: Dataframe with timestamps
    :type ratings: pandas.DataFrame
    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    """
    rt_global = (
        ratings[['timestamp']]
        .apply(get_reltime).timestamp.fillna(0.5).values)
    rt_user = (
        ratings.groupby('user_id').timestamp
        .apply(get_reltime).fillna(0.5).values)
    rt_item = (
        ratings.groupby('item_id').timestamp
        .apply(get_reltime).fillna(0.5).values)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['cf_data']['time']['r_global'] = rt_global
        f['cf_data']['time']['r_user'] = rt_user
        f['cf_data']['time']['r_item'] = rt_item


def process_rating_data(raw_path, data_path):
    """Creates id mappings, rating and time matrices

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path to save the processed data (.hdf5 file)
    :type data_path: str
    """
    ratings = _read_ratings(raw_path)
    ratings = _process_id_mappings(ratings, data_path)
    _create_rating_matrix(ratings, data_path)
    _create_time_matrices(ratings, data_path)


def _remove_and_order_users(df_user, data_path):
    """Remove users with no rating information and order them

    :param df_user: User dataframe with id
    :type df_user: pandas.DataFrame
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    :return: User dataframe without new users
    :rtype: pandas.DataFrame
    """
    with open(os.path.join(data_path, "dict_user_ar.pickle"), "rb") as f:
        dict_user_ar = pickle.load(f)
    df_user = df_user.set_index('id')
    df_user = df_user.reindex(dict_user_ar.keys(), fill_value=0)
    return df_user


def _add_occupation(df_user, raw_path):
    """Read and add occupation information to user table.

    :param df_user: User dataframe with id, age, gender and occupation
    :type df_user: pandas.DataFrame
    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :return: User dataframe with occupation information
    :rtype: pandas.DataFrame
    """
    df_occ = pd.read_csv(
        os.path.join(raw_path, 'u.occupation'),
        header=None, names=['occupation']
    )
    df_occ = pd.DataFrame(df_occ).reset_index()
    df_occ = df_occ.rename(columns={'index': 'occ_int'})
    df_user = df_user.merge(df_occ, on='occupation')
    del df_user['occupation']
    df_user = pd.get_dummies(df_user, columns=['occ_int'])
    df_user.iloc[:, 3:] /= np.sqrt(2)
    return df_user


def preprocess_user_features(raw_path, data_path):
    """Read and preprocess user features. It uses the age, gender
    and occupation.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    colnames = ['id', 'age', 'gender', 'occupation', 'zipcode']
    df_user = pd.read_csv(
        os.path.join(raw_path, 'u.user'),
        sep=r'|', header=None, names=colnames)
    del df_user['zipcode']
    df_user = _remove_and_order_users(df_user, data_path)
    df_user = _add_occupation(df_user, raw_path)
    df_user['gender'] = df_user.gender.map({'M': 0, 'F': 1})
    df_user['age'] = (df_user.age.astype(int) - 1) / 55
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']['user_fts'] = df_user.values


def _remove_and_order_items(df_item, data_path):
    """Remove items with no rating information and order them

    :param df_item: Item dataframe with id
    :type df_item: pandas.DataFrame
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    :return: User dataframe without new items
    :rtype: pandas.DataFrame
    """
    with open(os.path.join(data_path, "dict_item_ar.pickle"), "rb") as f:
        dict_item_ar = pickle.load(f)
    df_item = df_item.set_index('id')
    df_item = df_item.reindex(dict_item_ar.keys(), fill_value=0)
    return df_item


def preprocess_item_features(raw_path, data_path):
    """Read and preprocess item features. It uses the year and genre.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    colnames = [
        'id', 'title', 'rel_date', 'video release date', 'IMDb URL',
        'unknown', 'Action', 'Adventure', 'Animation', 'Childrens',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv(
        os.path.join(raw_path, 'u.item'),
        sep=r'|', header=None, names=colnames, engine='python')
    genre_headers = list(df_item.columns.values[6:])
    df_item = _remove_and_order_items(df_item, data_path)
    # extract year
    df_item['year_norm'] = [
        int(x[-4:]) / 10 if isinstance(x, str)
        else 1995 / 10 for x in df_item.rel_date]
    assert not (df_item.year_norm < (1800 / 10)).any()
    item_fts = df_item[['year_norm'] + genre_headers].values
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']["item_fts"] = item_fts


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
    os.makedirs(save_path)
    raw_path = os.path.join(args.raw_path, DATASET)
    colnames = ['u', 'v', 'r', 't']
    df_test = pd.read_csv(
        os.path.join(raw_path, f'u{args.split_id}.test'),
        sep='\t', header=None, names=colnames, engine='python')
    with open(os.path.join(args.data_path, "dict_user_ar.pickle"), "rb") as f:
        dict_user_ar = pickle.load(f)
    with open(os.path.join(args.data_path, "dict_item_ar.pickle"), "rb") as f:
        dict_item_ar = pickle.load(f)
    del df_test['r']
    del df_test['t']
    df_test['u'] = df_test.u.map(dict_user_ar)
    df_test['v'] = df_test.v.map(dict_item_ar)
    test_inds = df_test.sort_values(['u', 'v']).values
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
    with h5py.File(os.path.join(save_path, "train_mask.h5"), "w") as f:
        f["train_mask"] = train_mask

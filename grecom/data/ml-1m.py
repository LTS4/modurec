import os
import pickle

import h5py
import numpy as np
import pandas as pd
from torch_geometric.data import download_url, extract_zip

from grecom.data.utils import get_reltime
from grecom.data.geometric import create_similarity_graph

import stanfordnlp
import re
import string

DATASET = 'ml-1m'
URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'


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
        os.path.join(raw_path, 'ratings.dat'),
        sep='::', header=None, names=colnames, engine='python')
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


def preprocess_user_features(raw_path, data_path):
    """Read and preprocess user features. It uses the age, gender
    and occupation.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    colnames = ['id', 'gender', 'age', 'occupation', 'zipcode']
    df_user = pd.read_csv(
        os.path.join(raw_path, 'users.dat'),
        sep=r'::', header=None, names=colnames, engine='python')
    del df_user['zipcode']
    df_user = _remove_and_order_users(df_user, data_path)
    df_user['gender'] = df_user.gender.map({'M': 0, 'F': 1})
    df_user['age'] = (df_user.age.astype(int) - 1) / 55
    user_fts = np.zeros([len(df_user), 23])
    user_fts[:, :2] = df_user[['gender', 'age']].values
    for i, j in enumerate(df_user['occupation'].values):
        user_fts[i, 2 + int(j)] = 1 / np.sqrt(2)
    indices, weights = create_similarity_graph(user_fts)
    with h5py.File(os.path.join(data_path, "data.h5"), "a") as f:
        f['ca_data']['user_fts'] = user_fts
        f['ca_data']['user_graph_idx'] = indices
        f['ca_data']['user_graph_ws'] = weights


def _remove_and_order_items(ids, item_fts, data_path):
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
    sort_ids = np.argsort(ids)
    ids = ids[sort_ids]
    item_fts = item_fts[sort_ids, :]
    i_i = 0
    i_f = 0
    for key in dict_item_ar.keys():
        while key > ids[i_i] and i_i != len(ids):
            item_fts = np.delete(item_fts, i_f, axis=0)
            i_i += 1
        if i_i == len(ids):
            item_fts = np.append(
                item_fts, np.zeros(1, item_fts.shape[1]), axis=0)
        elif key == ids[i_i]:
            i_i += 1
            i_f += 1
        elif key < ids[i_i]:
            item_fts = np.insert(item_fts, i_f, 0, axis=0)
            i_f += 1
    return item_fts


def preprocess_item_features(raw_path, data_path):
    """Read and preprocess item features. It uses the year and genre.

    :param raw_path: Raw path (unprocessed data)
    :type raw_path: str
    :param data_path: Path of the processed data (.hdf5 file)
    :type data_path: str
    """
    all_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
    genres_dict = {g: i for i, g in enumerate(all_genres)}

    # For movie title
    if not os.path.exists('~/stanfordnlp_resources/en_ewt_models/en_ewt_tokenizer.pt'):
        stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
    vocab = set()
    title_words = []

    item_fts = []
    ids = []
    with open(os.path.join(raw_path, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            ids.append(int(id_))
            features = np.zeros(1 + len(all_genres))
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*\([0-9]{4}\)$', title)
            year = title[-5:-1]

            features[0] = int(year) / 10  # Year difference normalizer
            for g in genres_set:
                features[1 + genres_dict[g]] = 1 / np.sqrt(2)
            item_fts.append(features)

            # process title
            title = title[:-6].strip()
            doc = nlp(title)
            words = set()
            for s in doc.sentences:
                words.update(
                    w.lemma.lower() for w in s.words
                    if not re.fullmatch(
                        r'[' + string.punctuation + ']+', w.lemma))
            vocab.update(words)
            title_words.append(words)

    vocab = list(vocab)
    vocab_dict = {w: i for i, w in enumerate(vocab)}
    # bag-of-words
    for i, tw in enumerate(title_words):
        title_fts = np.zeros(len(vocab))
        title_fts[[vocab_dict[w] for w in tw]] = 1
        item_fts[i] = np.concatenate([item_fts[i], title_fts], axis=None)
    ids = np.array(ids)
    item_fts = np.vstack(item_fts)
    item_fts = _remove_and_order_items(ids, item_fts, data_path)
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

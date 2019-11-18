"""Movielens 1M preprocessing code from:
https://github.com/dmlc/dgl/blob/master/examples/pytorch/recommendation/rec/datasets/movielens.py
"""

import os
import re
import string

import numpy as np
import pandas as pd
import stanfordnlp
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from torch_geometric.data import download_url, extract_zip, Data
from torch_geometric.utils import to_undirected, subgraph
import torch


class RecommenderDataset(object):
    def __init__(self, args):
        # Dataset
        self.n_items = None
        self.n_users = None
        self.dict_user_ar = None
        self.dict_item_ar = None
        self.dict_user_ra = None
        self.dict_item_ra = None

        # Paths
        self.args = args
        self.raw_dir = os.path.join(args.raw_path, args.dataset)
        self.data_dir = os.path.join(args.data_path, args.dataset)

        # Download dataset if needed
        if not os.path.exists(self.raw_dir):
            self.url = self.select_url()
            self.download()

        # Create all dataframes
        self.users = self.preprocess_user_features()
        self.items = self.preprocess_item_features()
        self.ratings = self.create_dataframe_ratings()
        self.create_global_indices()

        self.rating_graph = self.create_rating_graph()

        # Create user and item similarity graphs
        self.user_graph = self.create_user_graph()
        self.item_graph = self.create_item_graph()
        self.similar_graph = self.create_similar_graph()

        self.rating_matrix, self.time_matrix = self.create_rating_matrices()

    def select_url(self):
        return {
            'ml-100k': 'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        }[self.args.dataset]

    def download(self):
        path = download_url(self.url, self.args.raw_path)
        extract_zip(path, self.args.raw_path)
        os.unlink(path)

    def create_global_indices(self):
        user_ids = np.unique(np.concatenate(
            [self.ratings.user_id.unique(), self.users.user_id.unique()]))
        item_ids = np.unique(np.concatenate(
            [self.ratings.item_id.unique(), self.items.item_id.unique()]))
        user_ids.sort()
        item_ids.sort()
        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        self.dict_user_ar = {id_: i for i, id_ in enumerate(user_ids)}  # ar = absolute -> relative
        self.dict_item_ar = {id_: i + self.n_users for i, id_ in enumerate(item_ids)}
        self.dict_user_ra = {i: id_ for i, id_ in enumerate(user_ids)}  # ra = relative -> absolute
        self.dict_item_ra = {i + self.n_users: id_ for i, id_ in enumerate(item_ids)}
        self.users['rel_id'] = [self.dict_user_ar[x] for x in self.users.user_id]
        self.items['rel_id'] = [self.dict_item_ar[x] for x in self.items.item_id]

    def create_user_graph(self, k=10):
        C = kneighbors_graph(np.stack(self.users.features.values), k, n_jobs=-1).toarray()
        D = kneighbors_graph(np.stack(self.users.features.values), k, n_jobs=-1, mode='distance').toarray()
        sigma = 1 / 3 * D[C == 1].mean()
        A = coo_matrix(C * np.exp(-D ** 2 / (2 * sigma ** 2)))
        row = [self.users.rel_id[x] for x in A.row]
        col = [self.users.rel_id[x] for x in A.col]
        return Data(edge_index=torch.tensor([row, col], dtype=torch.long),
                    edge_weight=torch.tensor(A.data, dtype=torch.float),
                    num_nodes=self.n_users,
                    ids=self.users.user_id.values,
                    x=torch.eye(self.n_users))
    
    def preprocess_user_features(self):
        """Reads the raw files and generates a feature vector for each user

        :return: Numpy array of size (num_users, num_features)
        :rtype: np.array
        """
        if self.args.dataset == 'ml-100k':
            return self.preprocess_user_features_ml100k()
        elif self.args.dataset == 'ml-1m':
            return self.preprocess_user_features_ml1m()

    def preprocess_user_features_ml100k(self):
        user_fts = []
        ids = []
        with open(os.path.join(self.raw_dir, 'u.user')) as f:
            for l in f:
                id_, age, gender, occupation, _ = l.strip().split('|')
                features = np.zeros(23)
                features[0] = 0 if gender == 'M' else 1
                features[1] = (int(age) - 1) / 55
                features[2 + int(occupation)] = 1 / np.sqrt(2)
                user_fts.append(features)
                ids.append(int(id_))
        return pd.DataFrame({'user_id': ids, 'features': user_fts})

    def preprocess_user_features_ml1m(self):
        user_fts = []
        ids = []
        with open(os.path.join(self.raw_dir, 'users.dat')) as f:
            for l in f:
                id_, gender, age, occupation, _ = l.strip().split('::')
                features = np.zeros(23)
                features[0] = 0 if gender == 'M' else 1
                features[1] = (int(age) - 1) / 55
                features[2 + int(occupation)] = 1 / np.sqrt(2)
                user_fts.append(features)
                ids.append(int(id_))
        return pd.DataFrame({'user_id': ids, 'features': user_fts})

    def create_item_graph(self, k=10):
        C = kneighbors_graph(np.stack(self.items.features.values), k, n_jobs=-1).toarray()
        D = kneighbors_graph(np.stack(self.items.features.values), k, n_jobs=-1, mode='distance').toarray()
        sigma = 1 / 3 * D[C == 1].mean()
        A = coo_matrix(C * np.exp(-D ** 2 / (2 * sigma ** 2)))
        row = [self.items.rel_id[x] for x in A.row]
        col = [self.items.rel_id[x] for x in A.col]
        return Data(edge_index=torch.tensor([row, col], dtype=torch.long),
                    edge_weight=torch.tensor(A.data, dtype=torch.float),
                    num_nodes=self.n_items,
                    ids=self.items.item_id.values,
                    x=torch.eye(self.n_items))

    def preprocess_item_features(self):
        """Reads the raw files and generates a feature vector for each item

        :return: Numpy array of size (num_items, num_features)
        :rtype: np.array
        """
        if self.args.dataset == 'ml-100k':
            return self.preprocess_item_features_ml100k()
        elif self.args.dataset == 'ml-1m':
            return self.preprocess_item_features_ml1m()

    def preprocess_item_features_ml100k(self):
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(os.path.join(self.raw_dir, 'u.item'), sep=r'|', header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]
        
        # extract year
        movie_df['year_norm'] = movie_df.title.str.slice(-5, -1).astype(int) / 10
        assert not (movie_df.year < (1800 / 10)).any()

        ids = movie_df['movie id'].astype(int).values
        item_fts = movie_df[['year_norm'] + genre_headers].values
        return pd.DataFrame({'item_id': ids, 'features': item_fts})

    def preprocess_item_features_ml1m(self):
        # Need to execute first "import standfordnlp; stanfordnlp.download('en', force=True)"
        all_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                      'Sci-Fi', 'Thriller', 'War', 'Western']
        genres_dict = {g: i for i, g in enumerate(all_genres)}

        # For movie title
        nlp = stanfordnlp.Pipeline(use_gpu=False, processors='tokenize,lemma')
        vocab = set()
        title_words = []

        item_fts = []
        ids = []
        with open(os.path.join(self.raw_dir, 'movies.dat'), encoding='latin1') as f:
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
        return pd.DataFrame({'item_id': ids, 'features': item_fts})

    def create_dataframe_ratings(self):
        ratings = []
        with open(os.path.join(self.raw_dir, 'ratings.dat')) as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [
                    int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'item_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                })
        ratings = pd.DataFrame(ratings)
        item_count = ratings['item_id'].value_counts()
        item_count.name = 'item_count'
        user_count = ratings['user_id'].value_counts()
        user_count.name = 'user_count'
        ratings = (
            ratings
            .join(item_count, on='item_id')
            .join(user_count, on='user_id'))
        return ratings

    def create_rating_graph(self):
        user_ids = [self.dict_user_ar[x] for x in self.ratings.user_id]
        item_ids = [self.dict_item_ar[x] for x in self.ratings.item_id]
        row = user_ids + item_ids
        col = item_ids + user_ids
        y = torch.tensor(self.ratings.rating.values, dtype=torch.float)
        return Data(edge_index=torch.tensor([row, col], dtype=torch.long),
                    num_nodes=self.n_items + self.n_users,
                    x=torch.eye(self.n_items + self.n_users),
                    y=torch.cat([y, y]))

    def create_similar_graph(self):
        return Data(edge_index=torch.cat((self.user_graph.edge_index, self.item_graph.edge_index), 1),
                    num_nodes=self.n_users + self.n_items,
                    x=torch.eye(self.n_items + self.n_users),
                    ids=np.concatenate([self.user_graph.ids, self.item_graph.ids]))

    def create_masks(self):
        n_edges = len(self.rating_graph.edge_index[0])
        assert n_edges % 2 == 0
        n_ratings = n_edges // 2
        train_mask, test_mask = train_test_split(
            np.array(range(n_ratings)), test_size=0.1, random_state=1)
        train_mask = np.concatenate([train_mask, train_mask + n_ratings])
        test_mask = np.concatenate([test_mask, test_mask + n_ratings])
        return train_mask, test_mask

    def create_toy_example(self):
        abs_item_ids = [1]  # Toy Story
        item_ids = [self.dict_item_ar[x] for x in abs_item_ids]
        # Add neighbor items
        item_edge_index = to_undirected(self.item_graph.edge_index)
        edge_ids = [
            i for i, x in enumerate(item_edge_index[0]) if x in item_ids
        ] + [i for i, x in enumerate(item_edge_index[1]) if x in item_ids]
        item_edge_index = item_edge_index[:, edge_ids]
        new_item_ids = item_edge_index.unique()
        # Add neighbor users
        edge_ids = [
            i for i, x in enumerate(self.rating_graph.edge_index[0])
            if x in item_ids]
        edge_ids += [
            i for i, x in enumerate(self.rating_graph.edge_index[1])
            if x in item_ids]
        rating_edge_index = self.rating_graph.edge_index[:, edge_ids]
        new_user_ids = rating_edge_index.unique()
        new_user_ids = new_user_ids[new_user_ids < self.n_users][:10]
        # Overwrite with subgraphs
        all_ids = torch.cat([new_user_ids, new_item_ids])

        self.similar_graph.edge_index = subgraph(
            all_ids, self.similar_graph.edge_index)[0]
        self.rating_graph.edge_index = subgraph(
            all_ids, self.rating_graph.edge_index)[0]
        df = self.ratings
        self.ratings = df.loc[
            df.item_id.isin(
                [self.dict_item_ra[x.item()] for x in new_item_ids]) &
            df.user_id.isin(
                [self.dict_user_ra[x.item()] for x in new_user_ids])]

    def create_rating_matrices(self):
        y = self.rating_graph.y
        assert len(y) % 2 == 0
        y = y[:len(y)//2]
        rating_index = np.array(self.rating_graph.edge_index)[:, :len(y)]
        rating_index[1, :] = rating_index[1, :] - self.n_users
        global_rt = (
            self.ratings[['timestamp']].apply(self._to_reltime)
            .timestamp.fillna(0.5).values)
        user_rt = (
            self.ratings.groupby('user_id').timestamp.apply(self._to_reltime)
            .fillna(0.5).values)
        item_rt = (
            self.ratings.groupby('item_id').timestamp.apply(self._to_reltime)
            .fillna(0.5).values)
        user_at = (
            self.ratings.groupby('user_id').timestamp.apply(self._to_abstime)
            .fillna(0).values)
        item_at = (
            self.ratings.groupby('item_id').timestamp.apply(self._to_abstime)
            .fillna(0).values)
        m_shape = (self.n_users, self.n_items)
        rating_matrix = coo_matrix((y, rating_index), shape=m_shape).toarray()
        time_matrix = np.stack([
            coo_matrix((global_rt, rating_index), shape=m_shape).toarray(),
            coo_matrix((user_rt, rating_index), shape=m_shape).toarray(),
            coo_matrix((item_rt, rating_index), shape=m_shape).toarray(),
            coo_matrix((user_at, rating_index), shape=m_shape).toarray(),
            coo_matrix((item_at, rating_index), shape=m_shape).toarray()])
        time_matrix = np.transpose(time_matrix, (1, 2, 0))
        return rating_matrix, time_matrix

    @staticmethod
    def _to_reltime(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def _to_abstime(x, scale=3600*24):
        """scale (seconds)"""
        return np.log(1 + (x - x.min())/scale)

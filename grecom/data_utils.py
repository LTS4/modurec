import os
import pickle

from grecom.data import RecommenderDataset


def create_recom_data(args, is_toy=False):
    path_data = os.path.join(args.data_path, 'recom_data.pkl')
    path_toy = os.path.join(args.data_path, 'toy_data.pkl')
    if (not is_toy) or (is_toy and not os.path.exists(path_toy)):
        if not os.path.exists(path_data):
            recom_data = RecommenderDataset(args, 'ml-1m')
            with open(path_data, 'wb') as f:
                pickle.dump(recom_data, f)
        else:
            with open(path_data, 'rb') as f:
                recom_data = pickle.load(f)
    if is_toy:
        if not os.path.exists(path_toy):
            recom_data.create_toy_example()
            with open(path_toy, 'wb') as f:
                pickle.dump(recom_data, f)
        else:
            with open(path_toy, 'rb') as f:
                recom_data = pickle.load(f)
    return recom_data

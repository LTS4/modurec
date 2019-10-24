import os
import pickle

import torch

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


def input_unseen_uv(train, val, preds, input_value=3):
    train_u, train_v = get_seen_uv(train)
    val_u, val_v = get_seen_uv(val)
    unseen_u = val_u - train_u
    unseen_u = torch.nonzero(unseen_u == 1)
    unseen_v = val_v - train_v
    unseen_v = torch.nonzero(unseen_v == 1)
    for i in range(len(preds)):
        preds[i][unseen_u, :] = input_value
        preds[i][:, unseen_v] = input_value
    return preds


def get_seen_uv(data):
    # [U, V]
    data_u = data.sum(1)
    data_v = data.sum(0)
    data_u[data_u > 0] = 1
    data_v[data_v > 0] = 1
    return data_u, data_v

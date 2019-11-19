import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from grecom.data_utils import input_unseen_uv
from grecom.layers import RecomConv, BilinearDecoder, GraphAutoencoder


class RecomNet(torch.nn.Module):
    def __init__(self, edge_sim, edge_rat, x, ratings, args):
        super(RecomNet, self).__init__()
        self.edge_sim = edge_sim
        self.edge_rat = edge_rat
        self.x = x
        self.ratings = ratings

        self.conv1 = RecomConv(x.shape[0], 200, args)
        self.conv2 = RecomConv(200, 200, args)
        self.decoder = BilinearDecoder(200, ratings.rating.mean(), args)

    def forward(self, mask):
        x = self.conv1(self.edge_sim, self.edge_rat, self.x)
        x = F.relu(x)
        x = self.conv2(self.edge_sim, self.edge_rat, x)
        h1 = x[self.edge_rat[0]]
        h2 = x[self.edge_rat[1]]
        pred = self.decoder(h1[mask], h2[mask])
        return pred


class GAENet(torch.nn.Module):
    """Graph Autoencoder Network
    """

    def __init__(self, recom_data, train_mask, val_mask, args, emb_size=500):
        super(GAENet, self).__init__()

        self.time_matrix = torch.tensor(
            recom_data.time_matrix, dtype=torch.float).to(args.device)
        self.features_u = torch.tensor(
            np.stack(recom_data.users.features.values), dtype=torch.float
            ).to(args.device)
        self.features_v = torch.tensor(
            np.stack(recom_data.items.features.values), dtype=torch.float
            ).to(args.device)

        self.item_ae = GraphAutoencoder(
            recom_data.n_users, args, emb_size,
            time_matrix=self.time_matrix.transpose(0, 1),
            feature_matrices=(self.features_v, self.features_u))
        self.user_ae = GraphAutoencoder(
            recom_data.n_items, args, emb_size,
            time_matrix=self.time_matrix,
            feature_matrices=(self.features_u, self.features_v))

        self.train_mask = torch.tensor(train_mask).to(args.device)
        self.val_mask = torch.tensor(val_mask).to(args.device)

        x = torch.tensor(recom_data.rating_matrix).to(args.device)
        self.x_train = (x * self.train_mask)
        self.x_val = (x * self.val_mask)

        self.mean_rating = self.x_train[self.x_train != 0].mean()
        self.mean_u, self.std_u = self.get_rating_statistics('user')
        self.mean_v, self.std_v = self.get_rating_statistics('item')

        self.x_u = ((x - self.mean_u) / self.std_u) * self.train_mask
        self.x_v = ((x - self.mean_v) / self.std_v) * self.train_mask
        self.x_m = (x - self.mean_rating) * self.train_mask

        self.edge_index_u = recom_data.user_graph.edge_index.to(args.device)
        self.edge_index_v = recom_data.item_graph.edge_index.to(
            args.device) - recom_data.n_users
        self.edge_weight_u = recom_data.user_graph.edge_weight.to(args.device)
        self.edge_weight_v = recom_data.item_graph.edge_weight.to(args.device)

        self.args = args

    def forward(self, mask=None, train='user', is_val=False):
        """mask: size 2*E
        """
        x_u = self.x_train
        x_v = self.x_train.T
        if is_val:
            p_u = self.user_ae(x_u, self.edge_index_u, self.edge_weight_u)
            p_v = self.item_ae(x_v, self.edge_index_v, self.edge_weight_v)
            p_u = nn.Hardtanh(1, 5)(p_u)
            p_v = nn.Hardtanh(1, 5)(p_v.T)
            p_u = input_unseen_uv(
                self.x_train, self.x_val, p_u, self.mean_rating)
            p_v = input_unseen_uv(
                self.x_train, self.x_val, p_v, self.mean_rating)
            return p_u, p_v
        else:
            real = self.x_train
            if train == 'user':
                if mask is not None:
                    real = self.x_train[mask, :]
                pred = self.user_ae(x_u, self.edge_index_u,
                                    self.edge_weight_u, mask=mask)
                reg_loss = self.user_ae.get_reg_loss()
            elif train == 'item':
                if mask is not None:
                    real = self.x_train[:, mask]
                pred = self.item_ae(x_v, self.edge_index_v,
                                    self.edge_weight_v, mask=mask)
                pred = pred.T
                reg_loss = self.item_ae.get_reg_loss()
            else:
                raise ValueError
            return real, pred, reg_loss

    def get_rating_statistics(self, norm):
        dim = (0 if norm == 'item' else 1)
        N = self.train_mask.sum(dim)
        mean = (self.x_train.sum(dim) / N)
        mean[N < 1] = 0
        mean = mean.unsqueeze(dim)
        std = (((self.x_train - mean) ** 2).sum(dim) / (N - 1)).sqrt()
        std[N < 2] = 1
        std = std.unsqueeze(dim)
        return mean, std

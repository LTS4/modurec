import torch
import torch.nn as nn
import torch.nn.functional as F

from grecom.layers import RecomConv, BilinearDecoder, RatingConv


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

    def __init__(self, recom_data, args):
        super(GAENet, self).__init__()
        self.recom_data = recom_data

        # User autoencoder
        self.user_enc = nn.Linear(recom_data.n_items, 200)
        self.user_conv = None
        self.user_dec = nn.Linear(200, recom_data.n_items)

        # Item autoencoder
        self.item_enc = nn.Linear(recom_data.n_users, 200)
        self.item_conv = None
        self.item_dec = nn.Linear(200, recom_data.n_users)

        self.args = args

    def forward(self, mask=None, val_mask=None):
        """mask: size 2*E
        """
        # Create input features
        N_u = self.recom_data.n_users
        N_v = self.recom_data.n_items
        N = N_u + N_v
        rat_mat = torch.sparse.LongTensor(
            torch.stack([x[mask] for x in self.recom_data.rating_graph.edge_index]),
            torch.FloatTensor(self.recom_data.rating_graph.y[mask]),
            torch.Size([N, N])
        ).to_dense().to(self.args.device)
        # x_u = rat_mat[:N_u, N_u:]
        x_v = rat_mat[N_u:, :N_u]
        h_v = nn.Sigmoid()(self.item_enc(x_v))
        p_v = self.item_dec(h_v)
        p_v = nn.Hardtanh(1, 5)(p_v)
        if val_mask is not None:
            val_mat = torch.sparse.LongTensor(
                torch.stack([x[val_mask] for x in self.recom_data.rating_graph.edge_index]),
                torch.FloatTensor(self.recom_data.rating_graph.y[val_mask]),
                torch.Size([N, N])
            ).to_dense().to(self.args.device)
            return val_mat[N_u:, :N_u], p_v
        return x_v, p_v

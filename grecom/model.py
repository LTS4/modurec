import torch
import torch.nn as nn
import torch.nn.functional as F

from grecom.layers import RecomConv, BilinearDecoder, RatingConv
import torch.nn.init as init
from torch.nn import Parameter
import math


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

    def __init__(self, recom_data, args, emb_size=200):
        super(GAENet, self).__init__()
        self.rating_index = recom_data.rating_graph.edge_index.to(args.device)
        self.y = recom_data.rating_graph.y.to(args.device)
        self.N_u = recom_data.n_users
        self.N_v = recom_data.n_items
        self.N = self.N_u + self.N_v

        # User autoencoder
        self.wenc_u = Parameter(torch.Tensor(emb_size, recom_data.n_items).to(args.device))
        self.benc_u = Parameter(torch.Tensor(emb_size).to(args.device))
        self.wdec_u = Parameter(torch.Tensor(recom_data.n_items, emb_size).to(args.device))
        self.bdec_u = Parameter(torch.Tensor(recom_data.n_items).to(args.device))

        # Item autoencoder
        self.wenc_v = Parameter(torch.Tensor(emb_size, recom_data.n_users).to(args.device))
        self.benc_v = Parameter(torch.Tensor(emb_size).to(args.device))
        self.wdec_v = Parameter(torch.Tensor(recom_data.n_users, emb_size).to(args.device))
        self.bdec_v = Parameter(torch.Tensor(recom_data.n_users).to(args.device))

        self.weights_list = [self.wenc_u, self.wdec_u, self.wenc_v, self.wdec_v]
        self.biases_list = [self.benc_u, self.bdec_u, self.benc_v, self.bdec_v]
        self.args = args

        self.reset_parameters()

    def reset_parameters(self):
        for w, b in zip(self.weights_list, self.biases_list):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)
    
    def forward(self, mask=None, val_mask=None):
        """mask: size 2*E
        """
        # Create input features
        print('Forward')
        rat_mat = torch.sparse.LongTensor(
            torch.stack([x[mask] for x in self.rating_index]),
            torch.FloatTensor(self.y[mask]),
            torch.Size([self.N, self.N])
        ).to_dense()
        # x_u = rat_mat[:N_u, N_u:]
        x_v = rat_mat[self.N_u:, :self.N_u]
        h_v = nn.Sigmoid()(F.linear(x_v, self.wenc_v, self.benc_v))
        p_v = F.linear(h_v, self.wdec_v, self.bdec_v)
        # p_v = nn.Hardtanh(1, 5)(p_v)
        if val_mask is not None:
            val_mat = torch.sparse.LongTensor(
                torch.stack([x[val_mask] for x in self.rating_index]),
                torch.FloatTensor(self.y[val_mask]),
                torch.Size([self.N, self.N])
            ).to_dense()
            return val_mat[self.N_u:, :self.N_u], p_v
        return x_v, p_v

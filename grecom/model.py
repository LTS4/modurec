import torch
import torch.nn as nn
import torch.nn.functional as F

from grecom.layers import RecomConv, BilinearDecoder
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

        self.x = torch.tensor(recom_data.rating_matrix).to(args.device)

        # Item autoencoder
        self.wenc = Parameter(torch.Tensor(emb_size, recom_data.n_users).to(args.device))
        self.benc = Parameter(torch.Tensor(emb_size).to(args.device))
        self.wdec = Parameter(torch.Tensor(recom_data.n_users, emb_size).to(args.device))
        self.bdec = Parameter(torch.Tensor(recom_data.n_users).to(args.device))

        self.weights_list = [self.wenc, self.wdec]
        self.biases_list = [self.benc, self.bdec]
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
        mask = torch.tensor(mask).to(self.args.device)
        # Create input features
        x_u = (self.x * mask)
        x_v = x_u.t()
        h_v = nn.Sigmoid()(F.linear(x_v, self.wenc, self.benc))
        p_v = F.linear(h_v, self.wdec, self.bdec)
        # p_v = nn.Hardtanh(1, 5)(p_v)
        if val_mask is not None:
            val_mask = torch.tensor(val_mask).to(self.args.device)
            val_u = (self.x * val_mask)
            val_v = val_u.t()
            return val_v, p_v
        return x_v, p_v

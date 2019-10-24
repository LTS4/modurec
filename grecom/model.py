import torch
import torch.nn as nn
import torch.nn.functional as F

from grecom.data_utils import input_unseen_uv
from grecom.layers import RecomConv, BilinearDecoder, GraphAutoencoder
from torch.nn import Parameter



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

        # Item autoencoder
        self.item_ae = GraphAutoencoder(recom_data.n_users, args, emb_size)
        self.user_ae = GraphAutoencoder(recom_data.n_items, args, emb_size)

        self.train_mask = torch.tensor(train_mask).to(args.device)
        self.val_mask = torch.tensor(val_mask).to(args.device)
        self.x = torch.tensor(recom_data.rating_matrix).to(args.device)
        self.x_train = (self.x * self.train_mask)
        self.x_val = (self.x * self.val_mask)

        self.args = args
    
    def forward(self, item_batch=None, is_val=False):
        """mask: size 2*E
        """
        # Create input features
        x = self.x_train
        if item_batch is not None:
            x = x[:, item_batch]
        # p_u = self.user_ae(x)
        p_v = self.item_ae(x.T).T
        if is_val:
            # p_u = nn.Hardtanh(1, 5)(p_u)
            p_v = nn.Hardtanh(1, 5)(p_v)
            p_v = input_unseen_uv(self.x_train, self.x_val, p_v)
            return self.x_val, p_v
        return x, p_v


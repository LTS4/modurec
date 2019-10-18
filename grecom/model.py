import torch
import torch.nn.functional as F
from torch.nn import Parameter

from grecom.layers import RecomConv, BilinearDecoder


class RecomNet(torch.nn.Module):
    def __init__(self, in_layers, edge_sim, edge_rat, x, ratings):
        super(RecomNet, self).__init__()
        self.edge_sim = edge_sim
        self.edge_rat = edge_rat
        self.x = x
        self.ratings = ratings

        self.conv1 = RecomConv(in_layers, 200)
        self.conv2 = RecomConv(200, 200)
        self.decoder = BilinearDecoder(200, ratings.rating.mean())

    def forward(self, mask):
        x = self.conv1(self.edge_sim, self.edge_rat, self.x)
        x = F.relu(x)
        x = self.conv2(self.edge_sim, self.edge_rat, x)
        h1 = x[self.edge_rat[0]]
        h2 = x[self.edge_rat[1]]
        pred = self.decoder(h1[mask], h2[mask])
        return pred


class AutographNet(torch.nn.Module):
    def __init__(self, in_layers, edge_sim, edge_rat, x, ratings):
        super(AutographNet, self).__init__()

    def forward(self, mask):
        pass




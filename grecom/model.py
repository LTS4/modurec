import torch
import torch.nn.functional as F
from torch.nn import Parameter

from grecom.layers import RecomConv


class RecomNet(torch.nn.Module):
    def __init__(self, in_layers):
        super(RecomNet, self).__init__()
        self.conv1 = RecomConv(in_layers, 200)
        self.conv2 = RecomConv(200, 200)

    def forward(self, edge_sim, edge_rat, x):
        x = self.conv1(edge_sim, edge_rat, x)
        x = F.relu(x)
        x = self.conv2(edge_sim, edge_rat, x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, emb_size, mean_rating):
        super(Decoder, self).__init__()
        self.Q = Parameter(torch.eye(emb_size))
        self.bias = Parameter(torch.Tensor([mean_rating]))

    def forward(self, h1, h2):
        return torch.sum((h1 @ self.Q) * h2, dim=1) + self.bias

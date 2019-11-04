import torch
import torch.sparse
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
import torch.nn.init as init
import math
import torch.nn as nn
import torch.nn.functional as F


def conv_norm(edge_index, num_nodes, edge_weight=None, dtype=None, symmetric=True):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                    device=edge_index.device)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    left_deg = deg_inv_sqrt[row] * edge_weight
    if symmetric:
        return left_deg * deg_inv_sqrt[col]  # symmetric norm
    return left_deg


class RecomConv(MessagePassing):
    def __init__(self, in_channels, out_channels, args):
        super(RecomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.weight_sim = xavier_normal_(Parameter(torch.Tensor(in_channels, out_channels).to(args.device)), gain=1)
        self.weight_rat = xavier_normal_(Parameter(torch.Tensor(in_channels, out_channels).to(args.device)), gain=1)
        self.weight_self = xavier_normal_(Parameter(torch.Tensor(in_channels, out_channels).to(args.device)), gain=1)

        self.bias = zeros_(Parameter(torch.Tensor(out_channels).to(args.device)))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args

    def forward(self, edge_sim, edge_rat, x):
        # TODO: improve performance by avoiding recalculation of normalizations
        x_self = torch.matmul(x, self.weight_self)
        x_sim = self.propagate(
            x=torch.matmul(x, self.weight_sim),
            edge_index=edge_sim,
            norm=conv_norm(edge_sim, x.size(0), None, x.dtype))
        x_rat = self.propagate(
            x=torch.matmul(x, self.weight_rat),
            edge_index=edge_rat,
            norm=conv_norm(edge_rat, x.size(0), None, x.dtype))
        return x_self + x_sim + x_rat + self.bias

    def message(self, x_j, edge_index, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class BilinearDecoder(torch.nn.Module):
    def __init__(self, emb_size, mean_rating, args):
        super(BilinearDecoder, self).__init__()
        self.Q = Parameter(torch.eye(emb_size).to(args.device))
        self.bias = Parameter(torch.Tensor([mean_rating]).to(args.device))

    def forward(self, h1, h2):
        return torch.sum((h1 @ self.Q) * h2, dim=1) + self.bias


class GraphConv0D(MessagePassing):
    def __init__(self, args):
        super(GraphConv0D, self).__init__(aggr='add')  # "Add" aggregation.
        self.weight = Parameter(torch.FloatTensor(1).to(args.device))

    def forward(self, x, edge_index, edge_weight=None, size=None):
        h = x * self.weight
        return self.propagate(edge_index, size=size, x=x, h=h,
                              norm=conv_norm(edge_index, x.size(0), edge_weight, x.dtype))

    def message(self, h_j, norm):
        return norm.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + x


class GraphConv1D(MessagePassing):
    def __init__(self, emb_size, args):
        super(GraphConv1D, self).__init__(aggr='add')  # "Add" aggregation.
        self.weight = Parameter(torch.FloatTensor(emb_size).to(args.device))

    def forward(self, x, edge_index, edge_weight=None, size=None):
        h = torch.matmul(x, torch.diag(self.weight))
        return self.propagate(edge_index, size=size, x=x, h=h,
                              norm=conv_norm(edge_index, x.size(0), edge_weight, x.dtype))

    def message(self, h_j, norm):
        return norm.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + x


class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_size, args, emb_size=500):
        super(GraphAutoencoder, self).__init__()
        self.wenc = Parameter(torch.Tensor(emb_size, input_size).to(args.device))
        self.benc = Parameter(torch.Tensor(emb_size).to(args.device))
        self.conv = GraphConv0D(args).to(args.device)
        #self.conv = GraphConv1D(emb_size, args).to(args.device)
        self.wdec = Parameter(torch.Tensor(input_size, emb_size).to(args.device))
        self.bdec = Parameter(torch.Tensor(input_size).to(args.device))

        self.dropout = nn.Dropout(p=0.7)
        self.dropout2 = nn.Dropout(p=0.5)

        self.weights_list = [self.wenc, self.wdec]
        self.biases_list = [self.benc, self.bdec]
        self.emb_size = emb_size
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        for w, b in zip(self.weights_list, self.biases_list):
            init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(b, -bound, bound)
        init.zeros_(self.conv.weight)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.dropout(x)
        x = F.linear(x, self.wenc, self.benc)
        x = nn.Sigmoid()(x)
        x = self.conv(x, edge_index, edge_weight)
        x = self.dropout2(x)
        p = F.linear(x, self.wdec, self.bdec)
        return p

    def get_reg_loss(self):
        return self.args.reg / 2 * (
            torch.norm(self.wenc) ** 2 +
            torch.norm(self.wdec) ** 2
        )


class TimeNN(torch.nn.Module):
    def __init__(self, args, emb_size=32):
        super(TimeNN, self).__init__()
        self.w_dense = Parameter(torch.Tensor(3).to(args.device))

        self.emb_size = emb_size
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_dense]:
            init.normal_(w, std=0.1)

    def forward(self, x):
        p = torch.matmul(x, self.w_dense)
        return p

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_dense) ** 2
        )

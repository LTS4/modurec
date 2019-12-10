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
    def __init__(self, input_size, args, emb_size=500,
                 time_matrix=None, feature_matrices=None):
        super(GraphAutoencoder, self).__init__()

        self.time_matrix = time_matrix
        self.feature_matrices = feature_matrices
        self.rating_add = Parameter(torch.FloatTensor(1).to(args.device))
        if time_matrix is not None:
            self.time_model = TimeNN(args, n_time_inputs=time_matrix.shape[-1])
            self.time_add = Parameter(torch.FloatTensor(1).to(args.device))
            self.time_mult = Parameter(torch.FloatTensor(1).to(args.device))
        if feature_matrices is not None:
            ft_sizes = [x.size(1) for x in feature_matrices]
            self.ft_model = FeatureNN(args, ft_sizes)
            self.ft1_mult = Parameter(torch.FloatTensor(1).to(args.device))
            self.ft2_mult = Parameter(torch.FloatTensor(1).to(args.device))
            self.ft_bias = Parameter(torch.FloatTensor(1).to(args.device))
        self.wenc = Parameter(torch.Tensor(emb_size, input_size).to(args.device))
        self.benc = Parameter(torch.Tensor(emb_size).to(args.device))
        self.conv = GraphConv0D(args).to(args.device)
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
        init.normal_(self.conv.weight, std=.01)
        init.zeros_(self.rating_add)
        if self.time_matrix is not None:
            init.normal_(self.time_mult, std=1e-4)
            init.normal_(self.time_add, std=1e-4)
        if self.feature_matrices is not None:
            init.normal_(self.ft1_mult, std=1e-4)
            init.normal_(self.ft2_mult, std=1e-4)
            init.zeros_(self.ft_bias, std=1e-4)

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.time_matrix is not None:
            x0 = x.clone()
            x = x0 * self.rating_add
            time_comp = self.time_model(self.time_matrix)
            x += (
                x0 * time_comp * self.time_mult +
                time_comp * self.time_add * (x0 > 0)
            )
        if self.feature_matrices is not None:
            fts = self.ft_model(self.feature_matrices)
            obs = (x != 0)
            A = F.sigmoid(
                self.ft1_mult * obs.sum(0).expand(x.size(0), -1) +
                self.ft2_mult * obs.sum(1).expand(-1, x.size(1)) +
                self.bias
            )
            A[obs.sum(0) == 0, obs.sum(1) == 0] = 0
            x = x * A + fts * (1 - A)
        x = self.dropout(x)
        x = F.linear(x, self.wenc, self.benc)
        x = nn.Sigmoid()(x)
        if not self.args.no_conv:
            x = self.conv(x, edge_index, edge_weight)
        x = self.dropout2(x)
        p = F.linear(x, self.wdec, self.bdec)
        return p

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.wenc) ** 2 +
            torch.norm(self.wdec) ** 2
        )
        if self.time_matrix is not None:
            reg_loss += self.time_model.get_reg_loss()
        if self.feature_matrices is not None:
            reg_loss += self.ft_model.get_reg_loss()
        return reg_loss


class TimeNN(torch.nn.Module):
    def __init__(self, args, emb_size=32, n_time_inputs=5):
        super(TimeNN, self).__init__()
        self.w_aff = Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.b_aff = Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.w_comb = Parameter(torch.Tensor(n_time_inputs).to(args.device))

        self.emb_size = emb_size
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.w_comb, self.w_aff, self.b_aff]:
            init.normal_(w, std=0.1)

    def forward(self, x):
        x = x * self.w_aff + self.b_aff
        x = nn.ReLU()(x)
        p = torch.matmul(x, self.w_comb)
        return p

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_comb) ** 2
        )
    
    def __repr__(self):
        return f"w_aff: {self.w_aff}, b_aff:{self.b_aff}, w_comb:{self.w_comb}"


class FeatureNN(torch.nn.Module):
    def __init__(self, args, ft_sizes, emb_size=500):
        super(FeatureNN, self).__init__()
        self.w_bilinear = Parameter(torch.FloatTensor(ft_sizes[0], ft_sizes[1]).to(args.device))
        self.bias = Parameter(torch.FloatTensor(1).to(args.device))

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.w_bilinear)
        init.zeros_(self.bias)

    def forward(self, vec_x):
        x_u, x_v = vec_x
        return x_u * self.w_bilinear * x_v.T + self.bias

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_bilinear) ** 2
        )
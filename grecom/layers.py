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
    def __init__(self, input_size, args, emb_size=500, feature_matrix=None, time_matrix=None, time_ndim=0, feature_ndim=0):
        super(GraphAutoencoder, self).__init__()

        # FiLM time
        self.time_matrix = time_matrix
        if time_matrix is not None:
            self.time_model = TimeNN(args)
            self.time_ndim = time_ndim
            film_size = {
                0: 1,
                1: input_size
            }[time_ndim]
            self.film_time = FiLMlayer(args, film_size)
            
        # FiLM features
        self.feature_matrix = feature_matrix
        if feature_matrix is not None:
            self.feature_model = FeatureNN(args, feature_matrix.size(1))
            self.feature_ndim = feature_ndim
            film_size = {
                0: 1,
                1: emb_size
            }[feature_ndim]
            self.film_fts = FiLMlayer(args, film_size)
        
        # Autoencoder
        self.wenc = Parameter(torch.Tensor(emb_size, input_size).to(args.device))
        self.benc = Parameter(torch.Tensor(emb_size).to(args.device))
        self.conv = GraphConv0D(args).to(args.device)
        #self.conv = GraphConv1D(emb_size, args).to(args.device)
        self.wdec = Parameter(torch.Tensor(input_size, emb_size).to(args.device))
        self.bdec = Parameter(torch.Tensor(input_size).to(args.device))
        self.wdec_clf = Parameter(torch.Tensor(5, emb_size, input_size).to(args.device))
        self.bdec_clf = Parameter(torch.Tensor(1, input_size, 5).to(args.device))

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

    def forward(self, x, edge_index, edge_weight=None, mask=None):
        if mask is not None:
            x = x[mask, :]
        if self.time_matrix is not None:
            if mask is not None:
                time_comp = self.time_model(self.time_matrix[mask,...])
            else:
                time_comp = self.time_model(self.time_matrix)
            x = self.film_time(x, time_comp, mask_add=(x > 0))
        x = self.dropout(x)
        x = F.linear(x, self.wenc, self.benc)
        x = nn.Sigmoid()(x)
        if self.feature_matrix is not None:
            fts_comp = self.feature_model(self.feature_matrix)
            x = self.film_fts(x, fts_comp)
        # x = self.conv(x, edge_index, edge_weight)
        x = self.dropout2(x)
        p_reg = F.linear(x, self.wdec, self.bdec)
        x = x.unsqueeze(0) @ self.wdec_clf
        x = x.permute(1, 2, 0) + self.bdec_clf
        p_clf = nn.Softmax(dim=2)(x)
        return p_reg, p_clf

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.wenc) ** 2 +
            torch.norm(self.wdec) ** 2
        )
        if self.time_matrix is not None:
            reg_loss += self.time_model.get_reg_loss()
        if self.feature_matrix is not None:
            reg_loss += self.feature_model.get_reg_loss()
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
        x = nn.ReLU()(x)  # Allows deactivation of some time inputs
        p = torch.matmul(x, self.w_comb)
        return p

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_comb) ** 2
        )
    
    def __repr__(self):
        return f"w_aff: {self.w_aff}, b_aff:{self.b_aff}, w_comb:{self.w_comb}"


class FeatureNN(torch.nn.Module):
    def __init__(self, args, ft_size, emb_size=500):
        super(FeatureNN, self).__init__()
        self.scale = Parameter(torch.FloatTensor(ft_size).to(args.device))
        self.wenc = Parameter(torch.FloatTensor(10, ft_size).to(args.device))
        self.benc = Parameter(torch.FloatTensor(10).to(args.device))
        self.wdec = Parameter(torch.FloatTensor(emb_size, 10).to(args.device))
        self.bdec = Parameter(torch.FloatTensor(emb_size).to(args.device))

        self.dropout = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.wenc)
        init.xavier_normal_(self.wdec)
        init.zeros_(self.benc)
        init.zeros_(self.bdec)
        init.ones_(self.scale)

    def forward(self, x):
        x = self.dropout(x)
        x = x * self.scale
        x = F.linear(x, self.wenc, self.benc)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = F.linear(x, self.wdec, self.bdec)
        return x

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.wenc) ** 2
            + torch.norm(self.wdec) ** 2
        )

class FiLMlayer(torch.nn.Module):
    def __init__(self, args, size):
        super(FiLMlayer, self).__init__()
        self.mult = Parameter(torch.FloatTensor(size).to(args.device))
        self.add = Parameter(torch.FloatTensor(size).to(args.device))
        self.other = Parameter(torch.FloatTensor(size).to(args.device))

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.other)
        init.normal_(self.mult, std=1e-4)
        init.normal_(self.add, std=1e-4)

    def forward(self, x, z, mask_add=1):
        return (x * z * self.mult) + z * self.add * mask_add + x * self.other

    def __repr__(self):
        return f"m|a|o: {self.mult.item()}|{self.add.item()}|{self.other.item()}"
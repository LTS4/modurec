import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing


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

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, dtype=None, symmetric=True):
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

    def forward(self, edge_sim, edge_rat, x):
        x_self = torch.matmul(x, self.weight_self)
        x_sim = self.propagate(
            x=torch.matmul(x, self.weight_sim),
            edge_index=edge_sim,
            norm=self.norm(edge_sim, x.size(0), None, x.dtype))
        x_rat = self.propagate(
            x=torch.matmul(x, self.weight_rat),
            edge_index=edge_rat,
            norm=self.norm(edge_rat, x.size(0), None, x.dtype))
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

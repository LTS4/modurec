import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from grecom.data.geometric import conv_norm
from torch_geometric.nn import MessagePassing


class TimeNN2Lbin(nn.Module): 

    def __init__(self, args, n_time_inputs=3, n_bins=2):
        super(TimeNN, self).__init__()
        dim_inp = n_time_inputs * n_bins
        self.n_bins = n_bins

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(dim_inp, 32)
        self.lin2 = nn.Linear(32, 1)

        self.args = args

    def forward(self, x):
        x = torch.floor(x * (self.n_bins - 1e-6)) # to map 1 to (n_bins - 1)
        x = F.one_hot(x.to(torch.long), num_classes=self.n_bins)
        x = x.view(x.shape[0], x.shape[1], -1).to(torch.float)
        x = self.relu(self.lin1(x))
        p = self.lin2(x)
        return torch.squeeze(p)


class TimeNN2Lq(nn.Module):

    def __init__(self, args, n_time_inputs=3):
        super(TimeNN, self).__init__()
        self.w_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.b_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(2 * n_time_inputs, 32)
        self.lin2 = nn.Linear(32, 1)

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.w_aff, std=0.1)
        init.normal_(self.b_aff, std=0.1)

    def forward(self, x):
        x = x * self.w_aff + self.b_aff
        x = self.relu(x)
        x = torch.cat([x, x**2], dim=2)
        x = self.relu(self.lin1(x))
        p = self.lin2(x)
        return torch.squeeze(p)

    def __repr__(self):
        return f"w_aff: {self.w_aff}, b_aff:{self.b_aff}"


class TimeNN(nn.Module): #TimeNN2L

    def __init__(self, args, n_time_inputs=3):
        super(TimeNN, self).__init__()
        self.w_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.b_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(n_time_inputs, 32)
        self.lin2 = nn.Linear(32, 1)

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.w_aff, std=0.1)
        init.normal_(self.b_aff, std=0.1)

    def forward(self, x):
        x = x * self.w_aff + self.b_aff
        x = self.relu(x)
        x = self.relu(self.lin1(x))
        p = self.lin2(x)
        return torch.squeeze(p)

    def __repr__(self):
        return f"w_aff: {self.w_aff}, b_aff:{self.b_aff}"


class TimeNN1L(nn.Module):
    """Takes the [U, V, t] tensor, calculates affine functions for each of the
    t [U, V] matrices and combines them linearly.
    """

    def __init__(self, args, n_time_inputs=3):
        super(TimeNN1L, self).__init__()
        self.w_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.b_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.w_comb = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))

        self.relu = nn.ReLU()
        self.lin = nn.Linear(n_time_inputs, 1)

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.w_aff, std=0.1)
        init.normal_(self.b_aff, std=0.1)
        init.normal_(self.w_comb, std=0.1)

    def forward(self, x):
        x = x * self.w_aff + self.b_aff
        x = self.relu(x)
        p = self.lin(x)
        return torch.squeeze(p)

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_comb) ** 2
        )

    def __repr__(self):
        return f"w_aff: {self.w_aff}, b_aff:{self.b_aff}, w_comb:{self.w_comb}"


class FilmLayer(nn.Module):
    """Combines two inputs with identical shape"""

    def  __init__(self, args):
        super(FilmLayer, self).__init__()
        self.add_x1 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.add_x2 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.mult_x = nn.Parameter(torch.FloatTensor(1).to(args.device))

        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.add_x1)
        init.normal_(self.add_x2, std=1e-4)
        init.normal_(self.mult_x, std=1e-4)

    def forward(self, x1, x2):
        """ x1 is the main input. It is modulated by x2."""
        x = x1 * self.add_x1 + x2 * self.add_x2 + x1 * x2 * self.mult_x
        return x


class ContentFiltering(torch.nn.Module):
    def __init__(self, args, ft_sizes, emb_size=500):
        super(ContentFiltering, self).__init__()
        self.w_bilinear = nn.Parameter(
            torch.FloatTensor(ft_sizes[0], ft_sizes[1]).to(args.device))
        self.bias = nn.Parameter(torch.FloatTensor(1).to(args.device))

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.w_bilinear)
        init.zeros_(self.bias)

    def forward(self, ft_x):
        return ft_x[0] @ self.w_bilinear @ ft_x[1].T + self.bias

    def get_reg_loss(self):
        return self.args.reg * (
            torch.norm(self.w_bilinear) ** 2
        )


class FeatureCombiner_scal(nn.Module):
    """Using the number of ratings as a variable, combines the feature and
    the rating representations."""

    def __init__(self, args):
        super(FeatureCombiner_scal, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.alpha)

    def forward(self, h, hf, ft_n):
        return h * self.alpha + hf * (1 - self.alpha)


class FeatureCombiner(nn.Module):
    """Using the number of ratings as a variable, combines the feature and
    the rating representations."""

    def __init__(self, args):
        super(FeatureCombiner, self).__init__()
        self.alpha_1 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.alpha_2 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.alpha_b = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.alpha_1)
        init.zeros_(self.alpha_2)
        init.zeros_(self.alpha_b)

    def forward(self, h, hf, ft_n):
        A = torch.sigmoid(
            100*self.alpha_1 * ft_n[0].view(-1, 1).expand(-1, len(ft_n[1])) +
            100*self.alpha_2 * ft_n[1].view(1, -1).expand(len(ft_n[0]), -1) +
            100*self.alpha_b
        )
        A_zeros = torch.ones_like(A)
        A_zeros[ft_n[0] == 0, :] = 0
        A_zeros[:, ft_n[1] == 0] = 0
        A = A * A_zeros
        return h * A + hf * (1 - A)


class FeatureNN2(nn.Module):
    """Using the number of ratings as a variable, combines the feature and
    the rating representations."""

    def __init__(self, args):
        super(FeatureNN2, self).__init__()
        self.alpha_1 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.alpha_b = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.alpha_1)
        init.zeros_(self.alpha_b)

    def forward(self, h, hf, ft_n):
        A = torch.sigmoid(
            100*self.alpha_1 * torch.unsqueeze(ft_n[0], 1) +
            100*self.alpha_b
        )
        A_zeros = torch.ones_like(A)
        A_zeros[ft_n == 0, :] = 0
        A = A * A_zeros
        return h * A + hf * (1 - A)


class GraphConv0D(MessagePassing):
    def __init__(self, args):
        super(GraphConv0D, self).__init__(aggr='add')  # "Add" aggregation.
        self.weight = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight, std=.01)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        h = x * self.weight
        norm = conv_norm(edge_index, x.size(0), edge_weight, x.dtype)
        return self.propagate(edge_index, size=size, x=x, h=h, norm=norm)

    def message(self, h_j, norm):
        return norm.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return aggr_out + x


class GraphConv0D_adaptive(MessagePassing):
    def __init__(self, args):
        super(GraphConv0D_adaptive, self).__init__(aggr='add')  # "Add" aggregation.
        self.alpha_1 = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.alpha_b = nn.Parameter(torch.FloatTensor(1).to(args.device))
        self.A = None
        self.reset_parameters()

    def reset_parameters(self):
        init.zeros_(self.alpha_1)
        init.zeros_(self.alpha_b)

    def forward(self, x, ft_n, edge_index, edge_weight=None, size=None):
        A = torch.sigmoid(
            100*self.alpha_1 * ft_n[0].view(-1, 1).expand(-1, x.shape[1]) +
            100*self.alpha_b
        )
        A_zeros = torch.ones_like(A)
        A_zeros[ft_n == 0, :] = 0
        self.A = A * A_zeros
        h = x
        norm = conv_norm(edge_index, x.size(0), edge_weight, x.dtype)
        return self.propagate(edge_index, size=size, x=x, h=h, norm=norm)

    def message(self, h_j, norm):
        return norm.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return x * self.A + aggr_out * (1 - self.A)
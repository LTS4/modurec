import torch
import torch.nn as nn
import torch.nn.init as init


class TimeNN(nn.Module):

    def __init__(self, args, n_time_inputs=3):
        super(TimeNN, self).__init__()
        self.w_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.b_aff = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))
        self.w_comb = nn.Parameter(torch.Tensor(n_time_inputs).to(args.device))

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.w_aff, std=0.1)
        init.normal_(self.b_aff, std=0.1)
        init.normal_(self.w_comb, std=0.1)

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


class FilmLayer(nn.Module):

    def __init__(self, args):
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
        x = x1 * self.add_x1 + x2 * self.add_x2 + x1 * x2 * self.mult_x
        return x


class FeatureNN(nn.Module):

    def __init__(self, args):
        super(FeatureNN, self).__init__()
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
            100*self.alpha_1 * torch.unsqueeze(ft_n, 1) +
            100*self.alpha_b
        )
        A_zeros = torch.ones_like(A)
        A_zeros[ft_n == 0, :] = 0
        #A_zeros[:, ft_n[1] == 0] = 0
        A = A * A_zeros
        return h * A + hf * (1 - A)

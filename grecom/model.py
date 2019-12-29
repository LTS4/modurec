import torch
import torch.nn as nn

from grecom.layers import TimeNN, FilmLayer, FeatureNN, GraphConv0D


class Autorec(nn.Module):
    requires_time = False
    requires_fts = False
    requires_graph = False

    def __init__(self, args, input_size, rating_range=(1, 5)):
        super(Autorec, self).__init__()
        self.args = args

        self.encoder = nn.Linear(input_size, 500).to(args.device)
        self.sig_act = nn.Sigmoid()
        self.decoder = nn.Linear(500, input_size).to(args.device)
        self.limiter = nn.Hardtanh(rating_range[0], rating_range[1])

    def forward(self, x):
        x = self.sig_act(self.encoder(x))
        p = self.decoder(x)
        if not self.training:
            p = self.limiter(p)
        return p

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.encoder.weight) ** 2 +
            torch.norm(self.decoder.weight) ** 2
        )
        return reg_loss


class AutorecPP(nn.Module):
    requires_time = True
    requires_fts = False
    requires_graph = False

    def __init__(self, args, input_size, rating_range=(1, 5)):
        super(AutorecPP, self).__init__()
        self.args = args

        self.time_nn = TimeNN(args, n_time_inputs=3)
        self.film_time = FilmLayer(args)
        self.dropout_input = nn.Dropout(0.7)
        self.encoder = nn.Linear(input_size, 500).to(args.device)
        self.sig_act = nn.Sigmoid()
        self.dropout_emb = nn.Dropout(0.5)
        self.decoder = nn.Linear(500, input_size).to(args.device)
        self.limiter = nn.Hardtanh(rating_range[0], rating_range[1])

    def forward(self, x, time_x):
        time_x = self.time_nn(time_x)
        x = self.film_time(x, time_x)
        x = self.dropout_input(x)
        x = self.sig_act(self.encoder(x))
        x = self.dropout_emb(x)
        p = self.decoder(x)
        if not self.training:
            p = self.limiter(p)
        return p

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.encoder.weight) ** 2 +
            torch.norm(self.decoder.weight) ** 2
        )
        reg_loss += self.time_nn.get_reg_loss()
        return reg_loss


class AutorecPPg(nn.Module):
    requires_time = True
    requires_fts = False
    requires_graph = True

    def __init__(self, args, input_size, rating_range=(1, 5)):
        super().__init__()
        self.args = args

        self.time_nn = TimeNN(args, n_time_inputs=3)
        self.film_time = FilmLayer(args)
        self.dropout_input = nn.Dropout(0.7)
        self.encoder = nn.Linear(input_size, 500).to(args.device)
        self.sig_act = nn.Sigmoid()
        self.conv = GraphConv0D(args).to(args.device)
        self.dropout_emb = nn.Dropout(0.5)
        self.decoder = nn.Linear(500, input_size).to(args.device)
        self.limiter = nn.Hardtanh(rating_range[0], rating_range[1])

    def forward(self, x, time_x, graph):
        graph = graph[0]
        time_x = self.time_nn(time_x)
        x = self.film_time(x, time_x)
        x = self.dropout_input(x)
        x = self.sig_act(self.encoder(x))
        x = self.conv(x, graph.edge_index, graph.edge_weight)
        x = self.dropout_emb(x)
        p = self.decoder(x)
        if not self.training:
            p = self.limiter(p)
        return p

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.encoder.weight) ** 2 +
            torch.norm(self.decoder.weight) ** 2
        )
        reg_loss += self.time_nn.get_reg_loss()
        return reg_loss


class AutorecPPP2(nn.Module):
    requires_time = True
    requires_fts = True
    requires_graph = False
    tr_steps = 3

    def __init__(self, args, input_size, ft_size, rating_range=(1, 5)):
        super(AutorecPPP2, self).__init__()
        self.args = args

        self.time_nn = TimeNN(args, n_time_inputs=3)
        self.film_time = FilmLayer(args)
        self.dropout_input = nn.Dropout(0.7)
        self.encoder = nn.Linear(input_size, 500).to(args.device)
        self.sig_act = nn.Sigmoid()
        self.dropout_emb = nn.Dropout(0.5)
        self.decoder = nn.Linear(500, input_size).to(args.device)
        self.limiter = nn.Hardtanh(rating_range[0], rating_range[1])

        self.ft_encoder1 = nn.Linear(ft_size[0], 1000).to(args.device)
        self.ft_encoder2 = nn.Linear(1000, 500).to(args.device)
        self.add_ft = FeatureNN(args)

    def forward(self, x, time_x, ft_x, ft_n):
        time_x = self.time_nn(time_x)
        h = self.film_time(x, time_x)
        h = self.dropout_input(h)
        h = self.sig_act(self.encoder(h))
        hf = self.ft_encoder(ft_x)
        h = self.add_ft(h, hf, ft_n)
        h = self.dropout_emb(h)
        p = self.decoder(h)
        if not self.training:
            p = self.limiter(p)
        return p

    def get_reg_loss(self):
        reg_loss = self.args.reg / 2 * (
            torch.norm(self.encoder.weight) ** 2 +
            torch.norm(self.decoder.weight) ** 2
        )
        reg_loss += self.time_nn.get_reg_loss()
        return reg_loss

    def train_step(self, step, **kwargs):
        return getattr(self, f'train_step{step}')(**kwargs)

    def train_step1(self, x, time_x, ft_x, ft_n):
        time_x = self.time_nn(time_x)
        h = self.film_time(x, time_x)
        h = self.dropout_input(h)
        h = self.sig_act(self.encoder(h))
        h = self.dropout_emb(h)
        p = self.decoder(h)
        if not self.training:
            p = self.limiter(p)
        return x, p

    def train_step2(self, x, time_x, ft_x, ft_n):
        time_x = self.time_nn(time_x)
        h = self.film_time(x, time_x)
        h = self.sig_act(self.encoder(h))
        hf = self.sig_act(self.ft_encoder1(ft_x[0]))
        hf = self.sig_act(self.ft_encoder2(hf))
        return h, hf

    def train_step3(self, x, time_x, ft_x, ft_n):
        time_x = self.time_nn(time_x)
        x0 = self.film_time(x, time_x)
        h = self.sig_act(self.encoder(x0))
        hf = self.sig_act(self.ft_encoder1(ft_x[0]))
        hf = self.sig_act(self.ft_encoder2(hf))
        h = self.add_ft(h, hf, (x0 != 0).sum(1))
        p = self.decoder(h)
        if not self.training:
            p = self.limiter(p)
        return x, p

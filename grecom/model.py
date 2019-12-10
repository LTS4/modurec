import torch
import torch.nn as nn

from grecom.layers import TimeNN, FilmLayer


class AutorecPP(nn.Module):
    def __init__(self, args, input_size, rating_range=(1, 5)):
        super(AutorecPP, self).__init__()
        self.requires_time = True
        self.args = args
        
        self.time_nn = TimeNN(args, n_time_inputs=3)
        self.film_time = FilmLayer(args)
        self.dropout_input = nn.Dropout(0.7)
        self.encoder = nn.Linear(input_size, 500).to(args.device)
        self.dropout_emb = nn.Dropout(0.5)
        self.decoder = nn.Linear(500, input_size).to(args.device)
        self.limiter = nn.Hardtanh(rating_range[0], rating_range[1])

    def forward(self, x, time_x):
        time_x = self.time_nn(time_x)
        x = self.film_time(x, time_x)
        x = self.dropout_input(x)
        h = self.encoder(x)
        x = self.dropout_emb(h)
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
     
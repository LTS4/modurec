from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import torch.nn.functional as F

from torch import optim
from tqdm import tqdm
import pandas as pd

from grecom.data import split_data, DataGenerator


def _get_input_sizes(data_gen):
    kwargs = {}
    kwargs['input_size'] = data_gen.data['input_size']
    if data_gen.use_fts:
        kwargs['ft_size'] = tuple(
            x.shape[1] for x in data_gen.data['fts']
        )
    return kwargs


def _get_model_kwargs(dd):
    kwargs = {}
    kwargs['x'] = dd['x'] * dd['train_mask']
    if 'time' in dd:
        kwargs['time_x'] = dd['time'] * dd['train_mask'].unsqueeze(-1)
    if 'fts' in dd:
        kwargs['ft_x'] = dd['fts']
        kwargs['ft_n'] = dd['counts']
    return kwargs


def train_model(args, model_class, rating_type):
    data_gen = DataGenerator(args, model_class, rating_type)
    input_sizes = _get_input_sizes(data_gen)
    model = model_class(args, **input_sizes)
    data = data_gen.next()
    ex = ThreadPoolExecutor()
    reg = pd.DataFrame({'epoch': [], 'tr_rmse': [], 'te_rmse': []})
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.96)
    for epoch in tqdm(range(args.epochs)):
        th_data = ex.submit(data_gen.next)
        x = data['x'] * data['train_mask']
        xt = data['x'] * (1 - data['train_mask'])
        optimizer.zero_grad()

        model.train()
        model_kwargs = _get_model_kwargs(data)
        p = model(**model_kwargs)
        loss = F.mse_loss(x[x != 0], p[x != 0]) + model.get_reg_loss()
        loss.backward()

        model.eval()
        p = model(**model_kwargs)
        reg = reg.append({
            'epoch': epoch,
            'tr_rmse': F.mse_loss(x[x != 0], p[x != 0]).item() ** (1/2),
            'te_rmse': F.mse_loss(xt[xt != 0], p[xt != 0]).item() ** (1/2)
        }, ignore_index=True)
        if (reg.te_rmse.min() == reg.iloc[-1].te_rmse):
            pred = p.clone()
        data = th_data.result()
        optimizer.step()
        scheduler.step()
    print(reg.tail(20))
    print(reg.te_rmse.min())
    return x, xt, pred


def train(args):
    model_module = import_module(f"grecom.model")
    Model = getattr(model_module, args.model)
    x, xt, pred_v = train_model(args, Model, 'I')
    _, _, pred_u = train_model(args, Model, 'U')
    p = (pred_u.T + pred_v) / 2
    print({
        'tr_rmse': F.mse_loss(x[x != 0], p[x != 0]).item() ** (1/2),
        'te_rmse': F.mse_loss(xt[xt != 0], p[xt != 0]).item() ** (1/2)
    })


def run_experiment(args):
    """Run experiment on testing scenario. Performs the following steps:
    - Split data according to scenario
    - Train model
    - Evaluate model and save results

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    split_data(args)
    train(args)

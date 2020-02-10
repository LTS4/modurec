from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
import os
from torch import optim
from tqdm import tqdm

from grecom.data import DataGenerator, split_data


def _get_input_sizes(data_gen):
    """Get parameters for model initialization.

    :param data_gen: Data generator with input size and feature information
    :type data_gen: DataGenerator
    :return: Dictionary with the parameters
    :rtype: dict
    """
    kwargs = {}
    kwargs['input_size'] = data_gen.data['input_size']
    kwargs['rating_range'] = data_gen.data['rating_range']
    if data_gen.use_fts:
        kwargs['ft_size'] = tuple(
            x.shape[1] for x in data_gen.data['ft_x']
        )
    return kwargs


def _get_model_kwargs(dd):
    """Get parameters for model training/evaluation.

    :param dd: Data batch returned by generator.
    :type dd: dict
    :return: Dictionary with the parameters
    :rtype: dict
    """
    kwargs = {}
    kwargs['x'] = dd['x'] * dd['train_mask']
    kwargs['ft_n'] = dd['ft_n']
    if 'time' in dd:
        kwargs['time_x'] = dd['time'] * dd['train_mask'].unsqueeze(-1)
    if 'ft_x' in dd:
        kwargs['ft_x'] = dd['ft_x']
    if 'graph' in dd:
        kwargs['graph'] = dd['graph']
    return kwargs


def train_model(args, model_class, rating_type):
    """Train and evaluate the model at each epoch.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    :param model_class: Model class object
    :type model_class: torch.nn.Module
    :param rating_type: 'I' for item ratings, 'U' for user ratings.
    :type rating_type: str
    :return: Train, test and prediction data
    :rtype: torch.Tensor
    """
    data_gen = DataGenerator(args, model_class, rating_type)
    input_sizes = _get_input_sizes(data_gen)
    model = model_class(args, **input_sizes).to(args.device)
    data = next(data_gen)
    ex = ThreadPoolExecutor()
    reg = pd.DataFrame({'epoch': [], 'tr_rmse': [], 'te_rmse': []})
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.96)
    # TODO: avoid np.concatenate, use len(data_gen.data['x'].data)
    all_x = np.array([])
    all_xt = np.array([])
    all_xind = np.empty((2, 0))
    all_xtind = np.empty((2, 0))
    for epoch in tqdm(range(args.epochs)):
        all_p = np.array([])
        all_pt = np.array([])
        data_gen.new_epoch = False
        while data_gen.new_epoch is False:
            th_data = ex.submit(data_gen.__next__)
            x = data['x'] * data['train_mask']
            xt = data['x'] * (1 - data['train_mask'])
            optimizer.zero_grad()

            model.train()
            model_kwargs = _get_model_kwargs(data)
            p = model(**model_kwargs)
            loss = F.mse_loss(x[x != 0], p[x != 0]) + model.get_reg_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                p = model(**model_kwargs)
                x_ind = torch.nonzero((x != 0), as_tuple=True)
                xt_ind = torch.nonzero((xt != 0), as_tuple=True)
                if epoch == 0:
                    all_x = np.concatenate([all_x, x[x_ind].cpu().numpy()])
                    all_xt = np.concatenate([all_xt, xt[xt_ind].cpu().numpy()])
                    all_xind = np.concatenate([
                        all_xind, torch.stack(x_ind, dim=0).cpu().numpy()], axis=1)
                    all_xtind = np.concatenate([
                        all_xtind, torch.stack(xt_ind, dim=0).cpu().numpy()],
                        axis=1)
                all_p = np.concatenate([all_p, p[x_ind].cpu().numpy()])
                all_pt = np.concatenate([all_pt, p[xt_ind].cpu().numpy()])
            data = th_data.result()
        reg = reg.append({
            'epoch': epoch,
            'tr_rmse': np.mean((all_x - all_p) ** 2) ** (1/2),
            'te_rmse': np.mean((all_xt - all_pt) ** 2) ** (1/2)
        }, ignore_index=True)
        if (reg.te_rmse.min() == reg.iloc[-1].te_rmse):
            pred = all_p.copy()
            pred_t = all_pt.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict()
            }, os.path.join(
                args.models_path, f"{rating_type}-{model_class.__name__}.tar"))
        scheduler.step()
    del data
    del model
    
    print(reg.tail(20))
    print(reg.te_rmse.min())
    return {
        'x': all_x, 'xt': all_xt, 'p': pred, 'pt': pred_t, 'ix': all_xind,
        'ixt': all_xtind
    }


def train_both_models(args, model_class):
    """Train, evaluate and combine both models at each epoch.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    :param model_class: Model class object
    :type model_class: torch.nn.Module
    :return: Train, test and prediction data
    :rtype: torch.Tensor
    """
    data_gen_i = DataGenerator(args, model_class, 'I')
    input_sizes = _get_input_sizes(data_gen_i)
    model_i = model_class(args, **input_sizes).to(args.device)
    data_i = next(data_gen_i)
    data_gen_u = DataGenerator(args, model_class, 'U')
    input_sizes = _get_input_sizes(data_gen_u)
    model_u = model_class(args, **input_sizes).to(args.device)
    data_u = next(data_gen_u)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, 
            list(model_u.parameters()) + list(model_i.parameters())), args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.96)
    init_epoch = 0
    ex = ThreadPoolExecutor()
    models_path = os.path.join(
        args.models_path, args.dataset, 
        f"{args.split_type}-{args.split_id}")
    os.makedirs(models_path, exist_ok=True)
    if args.warm_start:
        checkpoint = torch.load(
            os.path.join(models_path, f"B-{model_class.__name__}.tar"))
        model_i.load_state_dict(checkpoint['model_state_dict_i'])
        model_u.load_state_dict(checkpoint['model_state_dict_u'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['sched_state_dict'])
        reg = checkpoint['reg']
        reg_i = checkpoint['reg_i']
        reg_u = checkpoint['reg_u']
        init_epoch = int(reg.epoch.max()) + 1
    else:
        reg_i = pd.DataFrame({'epoch': [], 'tr_rmse': [], 'te_rmse': []})
        reg_u = reg_i.copy()
        reg = reg_i.copy()
    all_x_i = []
    all_xt_i = []
    all_x_u = []
    all_xt_u = []
    for epoch in tqdm(range(init_epoch, args.epochs)):
        # Item train and eval
        all_xind_i = []
        all_xtind_i = []
        all_p_i = []
        all_pt_i = []
        data_gen_i.new_epoch = False
        while data_gen_i.new_epoch is False:
            th_data = ex.submit(data_gen_i.__next__)
            x = data_i['x'] * data_i['train_mask']
            xt = data_i['x'] * (1 - data_i['train_mask'])
            optimizer.zero_grad()

            model_i.train()
            model_kwargs = _get_model_kwargs(data_i)
            p = model_i(**model_kwargs)
            loss = F.mse_loss(x[x != 0], p[x != 0]) + model_i.get_reg_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model_i.eval()
                p = model_i(**model_kwargs)
                x_ind = torch.nonzero((x != 0), as_tuple=True)
                xt_ind = torch.nonzero((xt != 0), as_tuple=True)
                all_p_i.append(p[x_ind].cpu().numpy())
                all_pt_i.append(p[xt_ind].cpu().numpy())
                if epoch == init_epoch:
                    all_x_i.append(x[x_ind].cpu().numpy())
                    all_xt_i.append(xt[xt_ind].cpu().numpy())
                x_ind = torch.stack(x_ind, dim=0)
                xt_ind = torch.stack(xt_ind, dim=0)
                x_ind[0] = data_i['patch_inds'][x_ind[0]]
                xt_ind[0] = data_i['patch_inds'][xt_ind[0]]
                all_xind_i.append(x_ind.cpu().numpy())
                all_xtind_i.append(xt_ind.cpu().numpy())
            data_i = th_data.result()
        all_xind_i = np.concatenate(all_xind_i, axis=1)
        all_xtind_i = np.concatenate(all_xtind_i, axis=1)
        ind_i = np.lexsort((all_xind_i[1, :], all_xind_i[0, :]))
        ind_it = np.lexsort((all_xtind_i[1, :], all_xtind_i[0, :]))
        all_p_i = np.concatenate(all_p_i)[ind_i]
        all_pt_i = np.concatenate(all_pt_i)[ind_it]
        if epoch == init_epoch:
            all_x = np.concatenate(all_x_i)[ind_i]
            all_xt = np.concatenate(all_xt_i)[ind_it]
        reg_i = reg_i.append({
            'epoch': epoch,
            'tr_rmse': np.mean((all_x - all_p_i) ** 2) ** (1/2),
            'te_rmse': np.mean((all_xt - all_pt_i) ** 2) ** (1/2)
        }, ignore_index=True)
        if (reg_i.te_rmse.min() == reg_i.iloc[-1].te_rmse):
            ex.submit(torch.save, *({
                'reg_i': reg_i,
                'model_state_dict_i': model_i.state_dict()
            }, os.path.join(models_path, f"I-{model_class.__name__}.tar")))
        # User train and eval
        all_xind_u = []
        all_xtind_u = []
        all_p_u = []
        all_pt_u = []
        data_gen_u.new_epoch = False
        while data_gen_u.new_epoch is False:
            th_data = ex.submit(data_gen_u.__next__)
            x = data_u['x'] * data_u['train_mask']
            xt = data_u['x'] * (1 - data_u['train_mask'])
            optimizer.zero_grad()

            model_u.train()
            model_kwargs = _get_model_kwargs(data_u)
            p = model_u(**model_kwargs)
            loss = F.mse_loss(x[x != 0], p[x != 0]) + model_u.get_reg_loss()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model_u.eval()
                p = model_u(**model_kwargs)
                x_ind = torch.nonzero((x != 0), as_tuple=True)
                xt_ind = torch.nonzero((xt != 0), as_tuple=True)
                all_p_u.append(p[x_ind].cpu().numpy())
                all_pt_u.append(p[xt_ind].cpu().numpy())
                x_ind = torch.stack(x_ind, dim=0)
                xt_ind = torch.stack(xt_ind, dim=0)
                x_ind[0] = data_u['patch_inds'][x_ind[0]]
                xt_ind[0] = data_u['patch_inds'][xt_ind[0]]
                all_xind_u.append(x_ind.cpu().numpy())
                all_xtind_u.append(xt_ind.cpu().numpy())
            data_u = th_data.result()
        all_xind_u = np.concatenate(all_xind_u, axis=1)
        all_xtind_u = np.concatenate(all_xtind_u, axis=1)
        ind_u = np.lexsort((all_xind_u[0, :], all_xind_u[1, :]))
        ind_ut = np.lexsort((all_xtind_u[0, :], all_xtind_u[1, :]))
        all_p_u = np.concatenate(all_p_u)[ind_u]
        all_pt_u = np.concatenate(all_pt_u)[ind_ut]
        reg_u = reg_u.append({
            'epoch': epoch,
            'tr_rmse': np.mean((all_x - all_p_u) ** 2) ** (1/2),
            'te_rmse': np.mean((all_xt - all_pt_u) ** 2) ** (1/2)
        }, ignore_index=True)
        if (reg_u.te_rmse.min() == reg_u.iloc[-1].te_rmse):
            ex.submit(torch.save, *({
                'reg_u': reg_u,
                'model_state_dict_u': model_u.state_dict()
            }, os.path.join(models_path, f"U-{model_class.__name__}.tar")))
        alpha = 0.5
        p = alpha*all_p_i + (1-alpha)*all_p_u
        pt = alpha*all_pt_i + (1-alpha)*all_pt_u
        reg = reg.append({
            'epoch': epoch,
            'tr_rmse': np.mean((all_x - p) ** 2) ** (1/2),
            'te_rmse': np.mean((all_xt - pt) ** 2) ** (1/2)
        }, ignore_index=True)
        if (reg.te_rmse.min() == reg.iloc[-1].te_rmse):
            ex.submit(torch.save, *({
                'reg': reg,
                'reg_i': reg_i,
                'reg_u': reg_u,
                'model_state_dict_i': model_i.state_dict(),
                'model_state_dict_u': model_u.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict()
            }, os.path.join(models_path, f"B-{model_class.__name__}.tar")))
        scheduler.step()
    res_i = reg_i.iloc[reg_i['te_rmse'].idxmin()]
    res_u = reg_u.iloc[reg_u['te_rmse'].idxmin()]
    res_b = reg.iloc[reg['te_rmse'].idxmin()]
    res = res_i.to_frame().T.append(res_u).append(res_b)
    res['rating_type'] = ['I','U','B']
    res['split_id'] = args.split_id
    print(res)
    return res


def train(args):
    model_module = import_module(f"grecom.model")
    Model = getattr(model_module, args.model)
    try:
        return train_both_models(args, Model)
    except (KeyboardInterrupt, SystemExit):
        models_path = os.path.join(
            args.models_path, args.dataset, 
            f"{args.split_type}-{args.split_id}")
        checkpoint = torch.load(
            os.path.join(models_path, f"B-{Model.__name__}.tar"))
        reg = checkpoint['reg']
        reg_i = checkpoint['reg_i']
        reg_u = checkpoint['reg_u']
        print('I:', reg_i.iloc[reg_i['te_rmse'].idxmin()])
        print('U:', reg_u.iloc[reg_u['te_rmse'].idxmin()])
        print('B:', reg.iloc[reg['te_rmse'].idxmin()])
    if False:
        res_i = train_model(args, Model, 'I')
        res_u = train_model(args, Model, 'U')
        ind_i = np.lexsort((res_i['ix'][1, :], res_i['ix'][0, :]))
        ind_it = np.lexsort((res_i['ixt'][1, :], res_i['ixt'][0, :]))
        ind_u = np.lexsort((res_u['ix'][0, :], res_u['ix'][1, :]))
        ind_ut = np.lexsort((res_u['ixt'][0, :], res_u['ixt'][1, :]))
        x = res_i['x'][ind_i]
        p = (res_i['p'][ind_i] + res_u['p'][ind_u]) / 2
        xt = res_i['xt'][ind_it]
        pt = (res_i['pt'][ind_it] + res_u['pt'][ind_ut]) / 2
        print({
            'tr_rmse': np.mean((x - p) ** 2) ** (1/2),
            'te_rmse': np.mean((xt - pt) ** 2) ** (1/2)
        })

def evaluate(args):
    """Evaluate and combine both models.

    :param args: Dictionary with execution arguments
    :type args: Namespace
    :param model_class: Model class object
    :type model_class: torch.nn.Module
    :return: Train, test and prediction data
    :rtype: torch.Tensor
    """
    model_module = import_module(f"grecom.model")
    model_class = getattr(model_module, args.model)
    data_gen_i = DataGenerator(args, model_class, 'I')
    input_sizes = _get_input_sizes(data_gen_i)
    model_i = model_class(args, **input_sizes).to(args.device)
    data_i = next(data_gen_i)
    data_gen_u = DataGenerator(args, model_class, 'U')
    input_sizes = _get_input_sizes(data_gen_u)
    model_u = model_class(args, **input_sizes).to(args.device)
    data_u = next(data_gen_u)

    ex = ThreadPoolExecutor()
    models_path = os.path.join(
        args.models_path, args.dataset, 
        f"{args.split_type}-{args.split_id}")
    checkpoint = torch.load(
        os.path.join(models_path, f"B-{model_class.__name__}.tar"))
    model_i.load_state_dict(checkpoint['model_state_dict_i'])
    model_u.load_state_dict(checkpoint['model_state_dict_u'])
    reg = checkpoint['reg']
    reg_i = checkpoint['reg_i']
    reg_u = checkpoint['reg_u']
    epoch = int(reg.epoch.max()) + 1
    all_x_i = []
    all_xt_i = []
    all_x_u = []
    all_xt_u = []
    # Item train and eval
    all_xind_i = []
    all_xtind_i = []
    all_p_i = []
    all_pt_i = []
    data_gen_i.new_epoch = False
    while data_gen_i.new_epoch is False:
        th_data = ex.submit(data_gen_i.__next__)
        x = data_i['x'] * data_i['train_mask']
        xt = data_i['x'] * (1 - data_i['train_mask'])

        model_kwargs = _get_model_kwargs(data_i)
        with torch.no_grad():
            model_i.eval()
            p = model_i(**model_kwargs)
            x_ind = torch.nonzero((x != 0), as_tuple=True)
            xt_ind = torch.nonzero((xt != 0), as_tuple=True)
            all_p_i.append(p[x_ind].cpu().numpy())
            all_pt_i.append(p[xt_ind].cpu().numpy())
            all_x_i.append(x[x_ind].cpu().numpy())
            all_xt_i.append(xt[xt_ind].cpu().numpy())
            x_ind = torch.stack(x_ind, dim=0)
            xt_ind = torch.stack(xt_ind, dim=0)
            x_ind[0] = data_i['patch_inds'][x_ind[0]]
            xt_ind[0] = data_i['patch_inds'][xt_ind[0]]
            all_xind_i.append(x_ind.cpu().numpy())
            all_xtind_i.append(xt_ind.cpu().numpy())
        data_i = th_data.result()
    all_xind_i = np.concatenate(all_xind_i, axis=1)
    all_xtind_i = np.concatenate(all_xtind_i, axis=1)
    ind_i = np.lexsort((all_xind_i[1, :], all_xind_i[0, :]))
    ind_it = np.lexsort((all_xtind_i[1, :], all_xtind_i[0, :]))
    all_p_i = np.concatenate(all_p_i)[ind_i]
    all_pt_i = np.concatenate(all_pt_i)[ind_it]
    all_x = np.concatenate(all_x_i)[ind_i]
    all_xt = np.concatenate(all_xt_i)[ind_it]
    reg_i = reg_i.append({
        'epoch': epoch,
        'tr_rmse': np.mean((all_x - all_p_i) ** 2) ** (1/2),
        'te_rmse': np.mean((all_xt - all_pt_i) ** 2) ** (1/2)
    }, ignore_index=True)
    # User train and eval
    all_xind_u = []
    all_xtind_u = []
    all_p_u = []
    all_pt_u = []
    data_gen_u.new_epoch = False
    while data_gen_u.new_epoch is False:
        th_data = ex.submit(data_gen_u.__next__)
        x = data_u['x'] * data_u['train_mask']
        xt = data_u['x'] * (1 - data_u['train_mask'])

        model_kwargs = _get_model_kwargs(data_u)
        with torch.no_grad():
            model_u.eval()
            p = model_u(**model_kwargs)
            x_ind = torch.nonzero((x != 0), as_tuple=True)
            xt_ind = torch.nonzero((xt != 0), as_tuple=True)
            all_p_u.append(p[x_ind].cpu().numpy())
            all_pt_u.append(p[xt_ind].cpu().numpy())
            x_ind = torch.stack(x_ind, dim=0)
            xt_ind = torch.stack(xt_ind, dim=0)
            x_ind[0] = data_u['patch_inds'][x_ind[0]]
            xt_ind[0] = data_u['patch_inds'][xt_ind[0]]
            all_xind_u.append(x_ind.cpu().numpy())
            all_xtind_u.append(xt_ind.cpu().numpy())
        data_u = th_data.result()
    all_xind_u = np.concatenate(all_xind_u, axis=1)
    all_xtind_u = np.concatenate(all_xtind_u, axis=1)
    ind_u = np.lexsort((all_xind_u[0, :], all_xind_u[1, :]))
    ind_ut = np.lexsort((all_xtind_u[0, :], all_xtind_u[1, :]))
    all_p_u = np.concatenate(all_p_u)[ind_u]
    all_pt_u = np.concatenate(all_pt_u)[ind_ut]
    reg_u = reg_u.append({
        'epoch': epoch,
        'tr_rmse': np.mean((all_x - all_p_u) ** 2) ** (1/2),
        'te_rmse': np.mean((all_xt - all_pt_u) ** 2) ** (1/2)
    }, ignore_index=True)
    alpha = 0.5
    p = alpha*all_p_i + (1-alpha)*all_p_u
    pt = alpha*all_pt_i + (1-alpha)*all_pt_u
    reg = reg.append({
        'epoch': epoch,
        'tr_rmse': np.mean((all_x - p) ** 2) ** (1/2),
        'te_rmse': np.mean((all_xt - pt) ** 2) ** (1/2)
    }, ignore_index=True)
    res_i = reg_i.iloc[reg_i['te_rmse'].idxmin()]
    res_u = reg_u.iloc[reg_u['te_rmse'].idxmin()]
    res_b = reg.iloc[reg['te_rmse'].idxmin()]
    res = res_i.to_frame().T.append(res_u).append(res_b)
    res['rating_type'] = ['I','U','B']
    res['split_id'] = args.split_id
    print(res)
    return res

def run_experiment(args):
    """Run experiment on testing scenario. Performs the following steps:
    - Split data according to scenario
    - Train model
    - Evaluate model and save results

    :param args: Dictionary with execution arguments
    :type args: Namespace
    """
    split_data(args)
    if args.eval:
        return evaluate(args)
    return train(args)

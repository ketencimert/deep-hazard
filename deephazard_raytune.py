# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:08:43 2022

@author: Mert
"""
import os
import argparse
import json
import numpy as np
import random

from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import torch
from torch.distributions.uniform import Uniform
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import SurvivalData, load_dataset
from models import LambdaNN
from utils import evaluate_model

def train(lambdann, optimizer, train_dataloader, importance_samples, device):

    epoch_losses = dict()
    tr_loglikelihood = - np.inf
    tr_loglikelihoods = []
    for (x, t, e) in train_dataloader:
        optimizer.zero_grad()

        importance_sampler = Uniform(0, t)
        t_samples = importance_sampler.sample((importance_samples,)).T

        train_loglikelihood = (
                lambdann(x=x, t=t).log().squeeze(-1) * e
                - torch.mean(
            lambdann(x=x, t=t_samples).view(x.size(0), -1),
            -1) * t
        ).mean()

        tr_loglikelihoods.append(train_loglikelihood.item())

        # minimize negative loglikelihood
        (-train_loglikelihood).backward()
        optimizer.step()
    tr_loglikelihood = np.mean(tr_loglikelihoods)
    epoch_losses['LL_train'] = tr_loglikelihood

    return epoch_losses

def train_deephazard(config):

    device = config['device'] #e.g. cuda:3 -> str
    dtype = config['dtype'] #e.g. float64 -> str

    random.seed(config['seed']) #e.g. 12345 -> int
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[dtype]

    #data is normalized unless specified otherwise
    outcomes, features = load_dataset(config['dataset']) #e.g. support -> str

    x, t, e = features, outcomes.time, outcomes.event
    n = len(features)
    tr_size = int(n * 0.7) #standard

    horizons = [0.25, 0.5, 0.75]

    times = np.quantile(t[e == 1], horizons).tolist()

    folds = np.array(list(range(config['cv_folds'])) * n)[:n] #e.g. 5 -> int

    #we are optimizing with respect to nth fold validation set of 5 folds
    #filter w.r.t. the fold
    x = features[folds != config['fold']]
    t = outcomes.time[folds != config['fold']]
    e = outcomes.event[folds != config['fold']]

    x_tr, x_val = x[:tr_size], x[tr_size:]
    t_tr, t_val = t[:tr_size], t[tr_size:]
    e_tr, e_val = e[:tr_size], e[tr_size:]

    et_tr = np.array(
        [
            (e_tr.values[i], t_tr.values[i]) for i in range(len(e_tr))
        ],
        dtype=[('e', bool), ('t', float)]
    )

    et_val = np.array(
        [
            (e_val.values[i], t_val.values[i]) for i in range(len(e_val))
        ],
        dtype=[('e', bool), ('t', float)]
    )

    train_data = SurvivalData(
        x_tr.values, t_tr.values, e_tr.values, device, dtype
    )
    valid_data = SurvivalData(
        x_val.values, t_val.values, e_val.values, device, dtype
    )

    train_dataloader = DataLoader(
        train_data, batch_size=config['bs'], shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=config['bs'], shuffle=False
    )

    #input size is fixed to x_tr size
    d_in = x_tr.shape[1]
    #output size is fixed to time until event size, which is 1
    d_out = 1
    d_hid = d_in // 2 if config['d_hid'] is None else config['d_hid']

    model = LambdaNN(
        d_in, d_out, d_hid,
        n_layers=config['n_layers'], p=config['p'],
        norm=config['norm'], activation=config['act']
        ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config['wd']
    )

    train_batch_size = train_dataloader.batch_size

    for _ in range(config['epochs']):
        epoch_losses = train(
            model.train(),
            optimizer,
            train_dataloader,
            config['imps'],
            device
            )
        valid_loglikelihood, cis, brs, roc_auc = evaluate_model(
            model.eval(), valid_dataloader, times,
            et_tr, et_val, config['imps'],
            train_batch_size, dtype, 
            device
            )
        epoch_losses['LL_valid'] = valid_loglikelihood
        for horizon in enumerate(horizons):
            epoch_losses[
                'C-Index {} quantile'.format(horizon[1])
            ] = cis[horizon[0]]
            epoch_losses[
                'Brier Score {} quantile'.format(horizon[1])
            ] = brs[0][horizon[0]]
            epoch_losses[
                'ROC AUC {} quantile'.format(horizon[1])
            ] = roc_auc[horizon[0]][0]

        # Set this to run Tune.
        tune.report(epoch_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #ARGS TO CHANGE
    parser.add_argument('--dataset', default='support', type=str)
    parser.add_argument('--fold', default=0, type=int)
    # ARGS TO KEEP FIXED:
    # dataset
    parser.add_argument('--cv_folds', default=5, type=int)
    # device args
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    # optimization args
    parser.add_argument('--dtype', default='float64', type=str)
    parser.add_argument('--lr', 
                        default=tune.choice([1e-4, 5e-4, 1e-3, 2e-3])
                        )
    parser.add_argument('--wd',         
                        default=tune.choice([1e-6, 5e-6, 1e-5, 5e-5, 1e-4])
                        )
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--bs',
                        default=tune.choice([64, 128, 256, 512])
                        )
    parser.add_argument('--imps',
                        default=tune.choice([64, 128, 256, 512])
                        )
    # model, encoder-decoder args
    parser.add_argument('--n_layers',
                        default=tune.choice([1, 2, 3, 4])
                        )
    parser.add_argument('--p',
                        default=tune.choice([1e-1, 2e-1, 3e-1, 4e-1, 5e-1])
                        )
    parser.add_argument('--d_hid', 
                        default=tune.choice([50, 100, 200, 300, 400])
                        )
    parser.add_argument('--act', 
                        default=tune.choice(['relu', 'elu', 'selu', 'silu'])
                        )
    parser.add_argument('--norm', default='layer')
    parser.add_argument('--save_metric', default='LL_valid', type=str)
    args = parser.parse_args()

    args.mode = 'max'
    if 'brier' in args.save_metric.lower():
        args.mode = 'min'

    config = vars(args)

    scheduler = ASHAScheduler(
            metric='_metric/'+config['save_metric'],  # this is validation loss for me
            mode=config['mode'],
            max_t=config['epochs'], # i set this to be relatively high
            grace_period=10,  # default here is `1`; increasing may help tuning runs
            reduction_factor=2, # default is `4`; should probably play with this as well
        )
    searcher = HyperOptSearch(
        metric='_metric/'+config['save_metric'],
        mode=config['mode']
        )
    reporter = CLIReporter(
        metric_columns=[config['save_metric'], "training_iteration"]
        )

    init(num_cpus=0, num_gpus=1)

    result = tune.run(
        train_deephazard,
        resources_per_trial={"cpu": 0, "gpu": 1},
        # resources_per_trial={"cpu": 1, "gpu": 0},
        config=config,
        num_samples=100,
        progress_reporter=reporter,
        scheduler=scheduler,
        search_alg=searcher,
        raise_on_failed_trial=False
        )

    config = result.get_best_config(
        metric='_metric/'+config['save_metric'],
        mode=config['mode']
        )

    os.makedirs('./tune_results', exist_ok=True)
    with open(
            './tune_results/{}_dha_fold_{}_tuned_parameters.json'.format(
                config['dataset'],
                config['fold']
                )
        , 'w') as f:
        json.dump(config, f)

    
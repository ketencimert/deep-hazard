# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:08:43 2022

@author: Mert
"""

import argparse
import numpy as np
import random
from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import MedianStoppingRule
import torch
from torch.distributions.uniform import Uniform
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from deephazard import *

class SurvivalData(torch.utils.data.Dataset):
    def __init__(self, x, t, e, device, dtype=torch.double):

        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(t, dtype=dtype),
                torch.tensor(e, dtype=dtype)
            ] for x, t, e in zip(x, t, e)
        ]

        self.device = device
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.device:
                self._cache[index][0] = self._cache[
                    index][0].to(self.device)

                self._cache[index][1] = self._cache[
                    index][1].to(self.device)

                self._cache[index][2] = self._cache[
                    index][2].to(self.device)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_

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


def test(
        model, batcher, quantiles, train, valid, 
        importance_samples, train_batch_size, dtype,
        device
        ):
    with torch.no_grad():

        times_tensor = torch.tensor(quantiles, dtype=dtype).to(device)
        times_tensor = times_tensor.unsqueeze(-1).repeat_interleave(
            train_batch_size, -1
        ).T

        importance_sampler = Uniform(0, times_tensor)
        t_samples_ = torch.transpose(
            importance_sampler.sample(
                (importance_samples,)
            ), 0, 1
        )

        loglikelihoods = []
        survival = []
        ts = []
        for (x, t, e) in batcher:
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample(
                (importance_samples,)
            ).T

            loglikelihood = (
                    model(x=x, t=t).log().squeeze(-1) * e
                    - torch.mean(
                model(x=x, t=t_samples).view(x.size(0), -1),
                -1) * t
            ).mean()

            loglikelihoods.append(loglikelihood.item())

            # For C-Index and Brier Score

            survival_quantile = []
            for i in range(len(quantiles)):
                int_lambdann = torch.mean(
                    model(
                        x=x,
                        t=t_samples_[:x.size(0), :, i]).view(x.size(0), -1),
                    -1) * quantiles[i]

                survival_quantile.append(torch.exp(-int_lambdann))

            survival_quantile = torch.stack(survival_quantile, -1)
            survival.append(survival_quantile)
            ts.append(t)

        ts = torch.cat(ts).cpu().numpy()
        survival = torch.cat(survival).cpu().numpy()
        risk = 1 - survival

        cis = []
        brs = []
        for i, _ in enumerate(quantiles):
            cis.append(
                concordance_index_ipcw(
                    train, valid, risk[:, i], quantiles[i]
                )[0]
            )

        brs.append(
            brier_score(
                train, valid, survival, quantiles
            )[1]
        )

        roc_auc = []
        for i, _ in enumerate(quantiles):
            roc_auc.append(
                cumulative_dynamic_auc(
                    train, valid, risk[:, i], quantiles[i]
                )[0]
            )

        return np.mean(loglikelihoods), cis, brs, roc_auc

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

    #we are optimizing with respect to first fold validation set of 5 folds
    fold = 0

    #filter w.r.t. first fold
    x = features[folds != fold]
    t = outcomes.time[folds != fold]
    e = outcomes.event[folds != fold]

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
        train_data, batch_size=config['batch_size'], shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=config['batch_size'], shuffle=False
    )

    #input size is fixed to x_tr size
    d_in = x_tr.shape[1]
    #output size is fixed to time until event size, which is 1
    d_out = 1
    d_hid = d_in // 2 if config['d_hid'] is None else config['d_hid']

    model = LambdaNN(
        d_in, d_out, d_hid,
        n_layers=config['n_layers'], p=config['dropout'],
        norm=config['norm'], activation=config['activation']
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
            config['importance_samples'],
            device
            )
        valid_loglikelihood, cis, brs, roc_auc = test(
            model.eval(), valid_dataloader, times,
            et_tr, et_val, config['importance_samples'],
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
    parser.add_argument('--batch_size',
                        default=tune.choice([64, 128, 256, 512])
                        )
    parser.add_argument('--importance_samples',
                        default=tune.choice([int(64), int(128), int(256), int(512)])
                        )
    # model, encoder-decoder args
    parser.add_argument('--n_layers',
                        default=tune.choice([1, 2, 3, 4])
                        )
    parser.add_argument('--dropout',
                        default=tune.choice([1e-1, 2e-1, 3e-1, 4e-1, 5e-1])
                        )
    parser.add_argument('--d_hid', 
                        default=tune.choice([50, 100, 200, 300, 400])
                        )
    parser.add_argument('--activation', 
                        default=tune.choice(['relu', 'elu', 'selu', 'silu'])
                        )
    parser.add_argument('--norm', default='layer')
    parser.add_argument('--save_metric', default='LL_valid', type=str)
    # dataset
    parser.add_argument('--dataset', default='flchain', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    args = parser.parse_args()

    args.mode = 'max'
    if 'brier' in args.save_metric.lower():
        args.mode = 'min'

    config = vars(args)

    mdn_scheduler = MedianStoppingRule(
        time_attr = 'time_total_s',
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
        scheduler=mdn_scheduler,
        raise_on_failed_trial = False)

    config = result.get_best_config(
        metric=config['save_metric'],
        mode=config['mode']
        )

    # print("Best config is:", results.get_best_result().config)
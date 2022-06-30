# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:12:27 2022

@author: Mert
"""
import argparse

from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

from auton_lab.auton_survival import datasets, preprocessing
from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
)

from models import LambdaNN, DeepHazardMixture
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd


def evaluate_model(model, batcher, quantiles, train, valid):
    with torch.no_grad():

        times_tensor = torch.tensor(quantiles, dtype=dtype).to(args.device)
        times_tensor = times_tensor.unsqueeze(-1).repeat_interleave(
            train_dataloader.batch_size, -1
        ).T

        importance_sampler = Uniform(0, times_tensor)
        t_samples_ = torch.transpose(
            importance_sampler.sample(
                (args.importance_samples,)
            ), 0, 1
        )

        loglikelihoods = []
        survival = []
        ts = []
        for (x, t, e) in batcher:
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample(
                (args.importance_samples,)
            ).T

            loglikelihood = [
                (
                        model(c=j, x=x, t=t).log().squeeze(-1) * e
                        - torch.mean(
                    model(c=j, x=x, t=t_samples).view(x.size(0), -1),
                    -1) * t
                )
                for j in range(args.mixture_size)
            ]

            loglikelihood = torch.stack(loglikelihood, -1)

            posterior = loglikelihood - loglikelihood.logsumexp(-1).view(-1, 1)
            posterior = posterior.exp()

            loglikelihood = torch.sum(
                loglikelihood.exp() * posterior, -1
            ).log()
            loglikelihood = loglikelihood.mean()
            loglikelihoods.append(loglikelihood.item())

            # For C-Index and Brier Score

            survival_quantile = []
            for i in range(len(times)):
                int_lambdann = [
                    torch.mean(
                        model(
                            c=j,
                            x=x,
                            t=t_samples_[:x.size(0), :, i]).view(x.size(0), -1),
                        -1) * times[i]
                    for j in range(args.mixture_size)
                ]

                loglikelihood = [
                    (
                            model(c=j, x=x, t=t).log().squeeze(-1) * e
                            - torch.mean(
                        model(c=j, x=x, t=t_samples).view(x.size(0), -1),
                        -1) * t
                    )
                    for j in range(args.mixture_size)
                ]

                loglikelihood = torch.stack(loglikelihood, -1)
                posterior = loglikelihood - loglikelihood.logsumexp(
                    -1
                ).view(-1, 1)
                posterior = posterior.exp()

                int_lambdann = torch.stack(int_lambdann, -1)
                int_lambdann = torch.sum(posterior * int_lambdann, -1)

                survival_quantile.append(torch.exp(-int_lambdann))

            survival_quantile = torch.stack(survival_quantile, -1)
            survival.append(survival_quantile)
            ts.append(t)

        ts = torch.cat(ts).cpu().numpy()
        survival = torch.cat(survival).cpu().numpy()
        risk = 1 - survival

        cis = []
        brs = []
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(
                    train, valid, risk[:, i], times[i]
                )[0]
            )

        brs.append(
            brier_score(
                train, valid, survival, times
            )[1]
        )

        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(
                cumulative_dynamic_auc(
                    train, valid, risk[:, i], times[i]
                )[0]
            )

        return np.mean(loglikelihoods), cis, brs, roc_auc


class SurvivalData(torch.utils.data.Dataset):
    def __init__(self, x, t, e, cuda, dtype=torch.double):

        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(t, dtype=dtype),
                torch.tensor(e, dtype=dtype)
            ] for x, t, e in zip(x, t, e)
        ]

        self.cuda = cuda
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.cuda:
                self._cache[index][0] = self._cache[
                    index][0].cuda(non_blocking=True)

                self._cache[index][1] = self._cache[
                    index][1].cuda(non_blocking=True)

                self._cache[index][2] = self._cache[
                    index][2].cuda(non_blocking=True)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # device args
    parser.add_argument('--device', default='cuda', type=str)
    # optimization args
    parser.add_argument('--dtype', default='float64', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--importance_samples', default=256, type=int)
    # model, encoder-decoder args
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--d_hid', default=200, type=int)
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--norm', default='layer', type=str)
    parser.add_argument('--mixture_size', default=3, type=int)
    parser.add_argument('--save_metric', default='LL_valid', type=str)
    # dataset
    parser.add_argument('--dataset', default='support', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    args = parser.parse_args()

    dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[args.dtype]

    outcomes, features = load_dataset(args.dataset)

    x, t, e = features, outcomes.time, outcomes.event
    n = len(features)
    tr_size = int(n * 0.7)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e == 1], horizons).tolist()

    fold_results = defaultdict(lambda: defaultdict(list))

    for fold in tqdm(range(args.cv_folds)):

        x = features[folds != fold]
        t = outcomes.time[folds != fold]
        e = outcomes.event[folds != fold]

        x_tr, x_val = x[:tr_size], x[tr_size:]
        t_tr, t_val = t[:tr_size], t[tr_size:]
        e_tr, e_val = e[:tr_size], e[tr_size:]

        x_te = features[folds == fold]
        t_te = outcomes.time[folds == fold]
        e_te = outcomes.event[folds == fold]

        et_tr = np.array(
            [
                (e_tr.values[i], t_tr.values[i]) for i in range(len(e_tr))
            ],
            dtype=[('e', bool), ('t', float)]
        )

        et_te = np.array(
            [
                (e_te.values[i], t_te.values[i]) for i in range(len(e_te))
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
            x_tr.values, t_tr.values, e_tr.values, args.device, dtype
        )
        valid_data = SurvivalData(
            x_val.values, t_val.values, e_val.values, args.device, dtype
        )
        test_data = SurvivalData(
            x_te.values, t_te.values, e_te.values, args.device, dtype
        )

        train_dataloader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False
        )
        test_dataloader = DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False
        )

        d_in = x_tr.shape[1]
        d_out = 1
        d_hid = d_in // 2 if args.d_hid is None else args.d_hid

        lambdanns = [LambdaNN(
            d_in, d_out, d_hid, args.n_layers, p=args.dropout,
            norm=args.norm, activation=args.activation
        ).to(args.device) for _ in range(args.mixture_size)
                     ]
        deephazardmixture = DeepHazardMixture(lambdanns)

        optimizer = optim.Adam(deephazardmixture.parameters(), lr=args.lr,
                               weight_decay=args.wd
                               )

        epoch_losses = defaultdict(list)

        epoch_tr_loglikelihoods = []
        epoch_val_loglikelihoods = []
        epoch_c_idxes = []

        tr_loglikelihood = - np.inf
        val_loglikelihood = - np.inf
        c_idx = - np.inf

        for epoch in range(args.epochs):

            print("Fold: {} Epoch: {}, LL_train: {}, LL_valid: {}".format(
                fold, epoch, tr_loglikelihood, val_loglikelihood)
            )

            deephazardmixture.train()
            tr_loglikelihoods = []
            for (x, t, e) in train_dataloader:
                optimizer.zero_grad()

                importance_sampler = Uniform(0, t)
                t_samples = importance_sampler.sample((args.importance_samples,)).T

                train_loglikelihood = [
                    (
                            deephazardmixture(c=i, x=x, t=t).log().squeeze(-1) * e
                            - torch.mean(
                        deephazardmixture(
                            c=i,
                            x=x,
                            t=t_samples
                        ).view(x.size(0), -1),
                        -1) * t
                    )
                    for i in range(args.mixture_size)
                ]

                train_loglikelihood = torch.stack(train_loglikelihood, -1)

                posterior = train_loglikelihood - train_loglikelihood.logsumexp(
                    -1
                ).view(-1, 1)
                posterior = posterior.exp()
                posterior = posterior.detach()

                elbo = torch.sum(train_loglikelihood * posterior, -1)
                elbo = elbo.mean()

                train_loglikelihood = torch.sum(
                    train_loglikelihood.exp() * posterior, -1
                ).log()

                train_loglikelihood = train_loglikelihood.mean()
                tr_loglikelihoods.append(train_loglikelihood.item())

                # minimize negative loglikelihood
                (-elbo).backward()
                optimizer.step()

            tr_loglikelihood = np.mean(tr_loglikelihoods)

            print("\nValidating Model...")
            # validate the model
            val_loglikelihood, cis, brs, roc_auc = evaluate_model(
                deephazardmixture.eval(), valid_dataloader, times, et_tr, et_val
            )

            epoch_losses['LL_train'].append(tr_loglikelihood)
            epoch_losses['LL_valid'].append(val_loglikelihood)

            for horizon in enumerate(horizons):
                print(f"For {horizon[1]} quantile,")
                print("TD Concordance Index:", cis[horizon[0]])
                print("Brier Score:", brs[0][horizon[0]])
                print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
                epoch_losses[
                    'C-Index {} quantile'.format(horizon[1])
                ].append(cis[horizon[0]])
                epoch_losses[
                    'Brier Score {} quantile'.format(horizon[1])
                ].append(brs[0][horizon[0]])
                epoch_losses[
                    'ROC AUC {} quantile'.format(horizon[1])
                ].append(roc_auc[horizon[0]][0])

            if epoch_losses[args.save_metric][-1] == max(
                    epoch_losses[args.save_metric]
            ):
                print("Caching Best Model...")
                best_lambdann = deepcopy(deephazardmixture)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 5))
        ax[0][0].plot(epoch_losses['LL_train'], color='b', label="LL_train")
        ax_twin = ax[0][0].twinx()
        ax_twin.plot(epoch_losses['LL_valid'], color='r', label="LL_valid")
        ax[0][0].legend(loc="center right")
        ax_twin.legend(loc="lower right")
        color = ['r', 'g', 'b']
        i = 0
        j = 0
        k = 0
        for (key, value) in epoch_losses.items():
            if 'C-Index' in key:
                ax[0][1].plot(value, color=color[i], label=key)
                ax[0][1].legend(loc="center right")
                i += 1
            elif 'Brier' in key:
                ax[1][0].plot(value, color=color[j], label=key)
                ax[1][0].legend(loc="upper left")
                j += 1
            elif 'ROC' in key:
                ax[1][1].plot(value, color=color[k], label=key)
                ax[1][1].legend(loc="center right")
                k += 1

        print("\nEvaluating Best Model...")
        test_loglikelihood, cis, brs, roc_auc = evaluate_model(
            best_lambdann.eval(), test_dataloader, times, et_tr, et_te
        )
        print("Test Loglikelihood: {}".format(test_loglikelihood))
        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index:", cis[horizon[0]])
            print("Brier Score:", brs[0][horizon[0]])
            print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

            fold_results[
                'Fold: {}'.format(fold)
            ][
                'C-Index {} quantile'.format(horizon[1])
            ].append(cis[horizon[0]])
            fold_results[
                'Fold: {}'.format(fold)
            ][
                'Brier Score {} quantile'.format(horizon[1])
            ].append(brs[0][horizon[0]])
            fold_results[
                'Fold: {}'.format(fold)
            ][
                'ROC AUC {} quantile'.format(horizon[1])
            ].append(roc_auc[horizon[0]][0])
        if args.cv_folds == 1:
            torch.save(best_lambdann, './saves/best_deephazardmixtures.pth')

    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
        ]
    fold_results.to_csv(
        './fold_results_{}_{}_n_mistures_{}.csv'.format(
            args.dataset, 
            'deep_hazard_mixtures',
            args.mixture_size
            )
    )
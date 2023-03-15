# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:12:27 2022

@author: Mert
"""
import os
import argparse

from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform
from tqdm import tqdm

from datasets import SurvivalData, load_dataset
from models import LambdaNN
from utils import evaluate_model

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', default='support', type=str,
                        help='dataset')
    parser.add_argument('--cv_folds', default=5, type=int, help='cv_folds')
    # device args
    parser.add_argument('--device', default='cuda', type=str)
    # optimization args
    parser.add_argument('--patience', default=800, type=float, help='patience')
    parser.add_argument('--dtype', default='float32', type=str, help='dtype')
    parser.add_argument('--lr', default=2e-3, type=float, help='learning_rate')
    parser.add_argument('--wd', default=1e-6, type=float, help='weight_decay')
    parser.add_argument('--epochs', default=4000, type=int, help='epochs')
    parser.add_argument('--bs', default=256, type=int, help='batch_size')
    parser.add_argument('--imps', default=512, type=int,
                        help='importance_samples')
    # model, encoder-decoder args
    parser.add_argument('--n_layers', default=2, type=int, help='n_layers')
    parser.add_argument('--only_shared', default=True)
    parser.add_argument('--p', default=0.4, type=float, help='dropout')
    parser.add_argument('--d_hid', default=400, type=int, help='d_hid')
    parser.add_argument('--act', default='relu', type=str, help='activation')
    parser.add_argument('--norm', default='layer', help='normalization')
    parser.add_argument('--save_metric', default='LL_valid', type=str,
                        help='save_metric')
    args = parser.parse_args()

    SEED = 12345
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    FLAGS = ', '.join(
        [
            str(y) + ' ' + str(x) for (y,x) in vars(args).items() if y not in [
                'device',
                'dataset',
                'cv_folds'
                ]
            ]
        )

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

    criterion  = [min if 'Brier' in args.save_metric else max][0]

    for fold in tqdm(range(args.cv_folds)):

        PATIENCE = 0
        STOP_REASON = 'END OF EPOCHS'

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
            train_data, batch_size=args.bs, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_data, batch_size=args.bs, shuffle=False
        )
        test_dataloader = DataLoader(
            test_data, batch_size=args.bs, shuffle=False
        )

        d_in = x_tr.shape[1]
        D_OUT = 1
        d_hid = d_in // 2 if args.d_hid is None else args.d_hid

        lambdann = LambdaNN(
            d_in, D_OUT, d_hid, args.n_layers, p=args.p,
            norm=args.norm, activation=args.act, dtype=dtype,
            only_shared=args.only_shared
        ).to(args.device)

        optimizer = optim.Adam(lambdann.parameters(), lr=args.lr,
                               weight_decay=args.wd
                               )

        epoch_results = defaultdict(list)

        epoch_tr_loglikelihoods = []
        epoch_val_loglikelihoods = []
        epoch_c_idxes = []

        tr_loglikelihood = - np.inf
        val_loglikelihood = - np.inf
        c_idx = - np.inf

        for epoch in range(args.epochs):

            print(
                "\nFold: {} Epoch: {}, LL_train: {}, LL_valid: {}".format(
                    fold,
                    epoch,
                    round(tr_loglikelihood, 6),
                    round(val_loglikelihood, 6),
                    )
                )

            lambdann.train()
            tr_loglikelihoods = []
            for (x, t, e) in train_dataloader:
                optimizer.zero_grad()

                importance_sampler = Uniform(0, t)
                t_samples = importance_sampler.sample((args.imps,)).T

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

            print("\nValidating Model...")
            # validate the model
            val_loglikelihood, cis, brs, roc_auc, _ = evaluate_model(
                lambdann.eval(), valid_dataloader, times, et_tr, et_val,
                args.bs, args.imps, dtype, args.device
            )

            epoch_results['LL_train'].append(tr_loglikelihood)
            epoch_results['LL_valid'].append(val_loglikelihood)

            for horizon in enumerate(horizons):
                print(f"For {horizon[1]} quantile,")
                print("TD Concordance Index:", cis[horizon[0]])
                print("Brier Score:", brs[0][horizon[0]])
                print("ROC AUC:", roc_auc[horizon[0]][0], "\n")
                epoch_results[
                    'C-Index {} quantile'.format(horizon[1])
                ].append(cis[horizon[0]])
                epoch_results[
                    'Brier Score {} quantile'.format(horizon[1])
                ].append(brs[0][horizon[0]])
                epoch_results[
                    'ROC AUC {} quantile'.format(horizon[1])
                ].append(roc_auc[horizon[0]][0])

            print('Patience: {}'.format(PATIENCE))

            if epoch_results[args.save_metric][-1] == max(
                    epoch_results[args.save_metric]
            ):
                print("Caching Best Model...")
                best_lambdann = deepcopy(lambdann)

            if 'Brier' in args.save_metric:
                if epoch_results[args.save_metric][-1] > criterion(
                        epoch_results[args.save_metric]
                        ):
                    PATIENCE += 1
                else:
                    PATIENCE = 0
            else:
                if epoch_results[args.save_metric][-1] < criterion(
                        epoch_results[args.save_metric]
                        ):
                    PATIENCE += 1
                else:
                    PATIENCE = 0
            if PATIENCE >= args.patience:
                print('Early Stopping...')
                STOP_REASON = 'EARLY STOP'
                break

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 5))
        ax[0][0].plot(epoch_results['LL_train'], color='b', label="LL_train")
        ax[0][0].plot(epoch_results['LL_valid'], color='r', label="LL_valid")
        ax[0][0].legend()
        ax[0][0].set_xlabel('Epochs')
        color = ['r', 'g', 'b']
        i = 0
        j = 0
        k = 0
        for (key, value) in epoch_results.items():
            if 'C-Index' in key:
                ax[0][1].plot(value, color=color[i], label=key)
                ax[0][1].legend()
                i += 1
            elif 'Brier' in key:
                ax[1][0].plot(value, color=color[j], label=key)
                ax[1][0].legend()
                j += 1
            elif 'ROC' in key:
                ax[1][1].plot(value, color=color[k], label=key)
                ax[1][1].legend()
                k += 1
        ax[0][1].set_xlabel('Epochs')
        ax[1][0].set_xlabel('Epochs')
        ax[1][1].set_xlabel('Epochs')
        plt.tight_layout()
        os.makedirs('./fold_figures', exist_ok=True)
        plt.savefig("./fold_figures/{}_fold_{}_{}_figs_({}).svg".format(
                args.dataset,
                fold,
                'dha',
                FLAGS
                )
            )

        epoch_results = pd.DataFrame(epoch_results)
        os.makedirs('./epoch_results', exist_ok=True)
        epoch_results.to_csv(
            './epoch_results/{}_fold_{}_{}_epoch_res_({}).csv'.format(
                args.dataset,
                fold,
                'dha',
                FLAGS
                )
            )

        print("\nEvaluating Best Model...")
        test_loglikelihood, cis, brs, roc_auc, ev = evaluate_model(
            best_lambdann.eval(), test_dataloader, times, et_tr, et_te,
            args.bs, args.imps, dtype, args.device, True
        )
        print("\nTest Loglikelihood: {}".format(test_loglikelihood))
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

                
        fold_results[
            'Fold: {}'.format(fold)
        ][
            'Integrated Brier Score'
        ].append(
            ev.brier_score(
                np.linspace(
                    min([x[1] for x in et_te]),
                    max([x[1] for x in et_te]), 
                    100
                    )
                ).mean()
        )
        fold_results[
            'Fold: {}'.format(fold)
        ][
            'Antolini C-Index'
        ].append(
            ev.concordance_td('antolini')
        )
        fold_results[
            'Fold: {}'.format(fold)
        ][
            'Integrated NBLL'
        ].append(
            ev.integrated_nbll(
                np.linspace(
                    min([x[1] for x in et_te]), 
                    max([x[1] for x in et_te]), 
                    100
                    )
                ).mean()
        )

        fold_results[
            'Fold: {}'.format(fold)
        ][
            'Stop Reason'
        ].append(STOP_REASON)

        os.makedirs('./model_checkpoints', exist_ok=True)
        torch.save(
            best_lambdann,
            './model_checkpoints/{}_fold_{}_{}_({}).pth'.format(
                args.dataset,
                fold,
                'dha',
                FLAGS
                )
            )

    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
        ]

    os.makedirs('./fold_results', exist_ok=True)
    fold_results.to_csv(
        './fold_results/{}_{}_fold_results_({}).csv'.format(
            args.dataset,
            'dha',
            FLAGS
            )
        )

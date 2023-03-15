# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:14:40 2022

@author: Mert
"""
import ast
import os
import operator

import argparse
from collections import defaultdict
import random

import numpy as np
import pandas as pd

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

import torch # For building the networks
import torchtuples as tt # Some useful functions
from tqdm import tqdm

from sklearn.model_selection import ParameterGrid
from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
)

from datasets import load_dataset

def train_model(nodes, layers, batch_norm, dropout, alpha, sigma, lr, wd,
                batch_size, **extras
                ):

    net = tt.practical.MLPVanilla(
        d,
        [[nodes]*layers],
        out_features,
        batch_norm,
        dropout,
        ).to(args.device)

    model = DeepHitSingle(
        net,
        tt.optim.Adam,
        alpha=alpha,
        sigma=sigma,
        duration_index=labtrans.cuts,
        device=args.device
        )

    model.optimizer.set_lr(lr)
    model.optimizer.param_groups[0]['weight_decay'] = wd

    callbacks = [tt.callbacks.EarlyStopping(patience=50)]
    logs = model.fit(
        x_train,
        y_train,
        batch_size,
        args.epochs,
        callbacks,
        val_data=val,
        verbose=False,
        )

    return model, logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Fixed parameters
    parser.add_argument('--dataset', default='metabric', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    # Tuned parameters
    parser.add_argument('--batch_size', default=[256, 512, 1024])
    parser.add_argument('--lr', default=[5e-4, 1e-3])
    parser.add_argument('--wd', default=[0, 1e-8, 1e-6, 1e-3, 1e-1])
    parser.add_argument('--batch_norm', default=[True])
    parser.add_argument('--layers', default=[1, 2, 3])
    parser.add_argument('--nodes', default=[128, 256, 512])
    parser.add_argument('--dropout', default=[
        0, 0.1, 0.4, 0.5
    ])
    parser.add_argument('--alpha', default=[
        1e-1, 2e-1, 4e-1, 8e-1, 1
    ])
    parser.add_argument('--sigma', default=[
        1e-1, 2.5e-1, 4e-1, 8e-1, 1, 2, 10
    ])
    parser.add_argument('--num_durations', default=[10, 50, 100])
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    INTERPOLATE = 100
    HYPERPARAMETER_SAMPLES = 300
    MODEL_NAME = 'deephit'

    fold_results = defaultdict(lambda: defaultdict(list))

    #1.LOAD DATASET
    outcomes, features = load_dataset(args.dataset)

    x, t, e = features, outcomes.time, outcomes.event
    n = len(features)
    train_size = int(n * 0.7)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e == 1], horizons).tolist()

    d = x.shape[1]

    params = ParameterGrid(
        {
            key:item for key,item in vars(args).items() if key not in [
                'dataset',
                'cv_folds',
                'epochs',
                'device',
                'seed'
                ]
            }
        )

    params = [
        params[_] for _ in np.random.choice(
            len(params),
            min(HYPERPARAMETER_SAMPLES, len(params)),
            replace=False
            )
        ]

    for fold in tqdm(range(args.cv_folds)):

        x = features[folds != fold].values.astype(np.float32)
        t = outcomes.time[folds != fold].values
        e = outcomes.event[folds != fold].values

        x_train, x_val = x[:train_size], x[train_size:]
        t_train, t_val = t[:train_size], t[train_size:]
        e_train, e_val = e[:train_size], e[train_size:]

        x_test = features[folds==fold].values.astype(np.float32)
        t_test = outcomes.time[folds==fold].values
        e_test = outcomes.event[folds==fold].values

        param_dict = dict()

        for param in tqdm(params, total=len(params)):

            labtrans = DeepHitSingle.label_transform(param['num_durations'])
            y_train = labtrans.fit_transform(t_train, e_train)
            y_val = labtrans.transform(t_val, e_val)

            train = (x_train, y_train)
            val = (x_val, y_val)

            in_features = x_train.shape[1]
            out_features = labtrans.out_features

            #2.DEFINE MODEL
            model, logs = train_model(**param)
            val_loss = min(logs._monitors['val_'].scores['loss']['score'])
            # surv = model.interpolate(INTERPOLATE).predict_surv_df(x_val)
            # ev = EvalSurv(surv, t_val, e_val, censor_surv='km')
            # param_dict[str(param)] = ev.concordance_td('antolini')
            param_dict[str(param)] = val_loss

        best_config = ast.literal_eval(
            min(
                param_dict.items(), key=operator.itemgetter(1)
                )[0]
            )

        model = train_model(**best_config)[0]

        surv = model.interpolate(INTERPOLATE).predict_surv_df(x_test)
        survival = []
        list_lookup = surv.index.tolist()
        for time in times:
            loc = min(
                range(len(list_lookup)),
                key=lambda i: abs(list_lookup[i]-time)
                )
            survival.append(surv.values[loc,:].reshape(-1,1))

        survival = np.concatenate(survival,-1)
        risk = 1 - survival

        et_train = np.array(
            [
                (e_train[i], t_train[i]) for i in range(len(e_train))
            ],
            dtype=[('e', bool), ('t', float)]
        )

        et_test = np.array(
            [
                (e_test[i], t_test[i]) for i in range(len(e_test))
            ],
            dtype=[('e', bool), ('t', float)]
        )

        cis = []
        brs = []
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(
                    et_train, et_test, risk[:, i], times[i]
                )[0]
            )

        max_val = max([k[1] for k in et_test])
        max_tr = max([k[1] for k in et_train])
        while max_val > max_tr:
            idx = [k[1] for k in et_test].index(max_val)
            et_test = np.delete(et_test, idx, 0)
            survival = np.delete(survival, idx, 0)
            risk = np.delete(risk, idx, 0)
            max_val = max([k[1] for k in et_test])

        brs.append(
            brier_score(
                et_train, et_test, survival, times
            )[1]
        )

        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(
                cumulative_dynamic_auc(
                    et_train, et_test, risk[:, i], times[i]
                )[0]
            )

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

        ev = EvalSurv(surv, t_test, e_test, censor_surv='km')
        fold_results[
            'Fold: {}'.format(fold)
            ][
                'Integrated Brier Score'
                ].append(
                    ev.brier_score(np.linspace(t.min(), t.max(), 100)
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
                np.linspace(t.min(), t.max(), 100)
                ).mean()
            )
                    
    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
            ]

    os.makedirs('./fold_results', exist_ok=True)
    fold_results.to_csv(
        './fold_results/fold_results_{}_{}_seed_{}.csv'.format(
            args.dataset,
            MODEL_NAME,
            SEED
            )
        )
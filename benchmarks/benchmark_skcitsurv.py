# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:14:40 2022

@author: Mert
"""
import os

import argparse
from collections import defaultdict

import numpy as np
import random
import torch

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_ipcw,
    brier_score,
    cumulative_dynamic_auc
    )

from tqdm import tqdm
import pandas as pd
from pycox.evaluation import EvalSurv

from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--model_name', default='coxph', type=str)
    parser.add_argument('--dataset', default='metabric', type=str)
    parser.add_argument('--seed', default=12345, type=int)
    args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    HYPERPARAMETER_SAMPLES = 100

    outcomes, features = load_dataset(args.dataset)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()

    unique_times = np.unique(outcomes['time'].values)

    n = len(features)
    train_size = int(n*0.7)

    folds = np.array(list(range(args.cv_folds))*n)[:n]
    np.random.shuffle(folds)

    fold_results = defaultdict(lambda: defaultdict(list))

    param_grid = {
        'survivalforest':
            {
                'max_depth': [None, 5],
                'n_estimators': [50, 100, 200],
                'max_features': [
                    int(features.shape[-1] ** (0.5)),
                    features.shape[-1] // 2,
                    features.shape[-1]
                ],
                'min_samples_split': [2, 10, 100, 200],
                'n_jobs': [-1]
            },
        'coxph':
            {
                'alpha': [0, 1e-3, 1e-2, 1e-1]
            }
    }[args.model_name]

    model_choice = {
        'survivalforest':RandomSurvivalForest,
        'coxph':CoxPHSurvivalAnalysis
        }[args.model_name]

    for fold in tqdm(range(args.cv_folds)):

        x = features[folds!=fold]
        t = outcomes.time[folds!=fold]
        e = outcomes.event[folds!=fold]

        x_train, x_val = x[:train_size], x[train_size:]
        t_train, t_val = t[:train_size], t[train_size:]
        e_train, e_val = e[:train_size], e[train_size:]

        x_test = features[folds==fold].astype(np.float32)
        t_test = outcomes.time[folds==fold]
        e_test = outcomes.event[folds==fold]

        et_train = np.array(
            [(e_train.values[i], t_train.values[i]) for i in range(len(e_train))],
                          dtype = [('e', bool), ('t', float)])
        et_test = np.array(
            [(e_test.values[i], t_test.values[i]) for i in range(len(e_test))],
                          dtype = [('e', bool), ('t', float)])
        et_val = np.array(
            [(e_val.values[i], t_val.values[i]) for i in range(len(e_val))],
                          dtype = [('e', bool), ('t', float)])
        
        x_train_val = np.concatenate([x_train, x_val], 0)
        et_train_val = np.concatenate([et_train, et_val], 0)
        
        split_index = np.concatenate(
            [-np.ones_like(x_train)[:,0],
             np.zeros_like(x_val)[:,0]]
            )

        model_ = GridSearchCV(
            estimator=model_choice(),
            param_grid=param_grid,
            cv=PredefinedSplit(split_index),
            n_jobs=-1
            )
        
        model_.fit(x_train_val, et_train_val)
        best_estimator = model_.best_estimator_.get_params()

        model = model_choice(**best_estimator)
        model.fit(x_train, et_train)

        surv = model.predict_survival_function(x_test, return_array=True)
        event_times = model.event_times_

        ev = EvalSurv(
            pd.DataFrame(surv.T, index=event_times),
            np.asarray([t[1] for t in et_test]),
            np.asarray([t[0] for t in et_test]),
            censor_surv='km'
            )

        survival = []
        for time in times:
            loc = min(
                range(len(event_times)),
                key=lambda i: abs(event_times[i]-time)
                )
            survival.append(surv[:,loc].reshape(-1,1))

        survival = np.concatenate(survival,-1)
        risk = 1 - survival

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
            args.model_name,
            SEED
            )
        )

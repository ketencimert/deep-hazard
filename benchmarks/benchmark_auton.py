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

from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset
from auton_lab.auton_survival.models.dsm import DeepSurvivalMachines
from auton_lab.auton_survival.models.cph import DeepCoxPH
from auton_lab.auton_survival.models.dcm import DeepCoxMixtures
from auton_lab.auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--model_name', default='dsm', type=str)
    parser.add_argument('--dataset', default='support', type=str)

    args = parser.parse_args()

    SEED = 12345
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    HYPERPARAMETER_SAMPLES = 300

    outcomes, features = (
        var.astype(np.float64) for var in load_dataset(args.dataset)
        )

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()

    unique_times = np.unique(outcomes['time'].values)

    n = len(features)
    tr_size = int(n*0.7)

    folds = np.array(list(range(args.cv_folds))*n)[:n]
    np.random.shuffle(folds)

    fold_results = defaultdict(lambda: defaultdict(list))

    param_grid = {
        'dsm': {
            'k' : [1, 2, 3, 4, 6, 8, 10],
            'distribution' : ['LogNormal', 'Weibull'],
            'learning_rate' : [1e-4, 5e-4, 1e-3],
            'nodes_' : [[64], [128], [256], [512]],
            'layers_': [1, 2, 3],
            'discount': [1/3, 1/2, 3/4, 1]
            },
        'cph': {
            'nodes_' : [[64], [128], [256], [512]],
            'layers_': [1, 2, 3],            
            },
        'dcm': {
            'k' : [1, 2, 3, 4, 6, 8, 10, 20],
            'nodes_' : [[64], [128], [256], [512]],
            'layers_': [1, 2, 3],
            'batch_size': [128, 256, 512, 1024]
            },
        'cmhe':{
            'k':[1,2,3,],
            'g':[1,2,3],
            'a':[],
            },
        }[args.model_name]

    params = ParameterGrid(param_grid)
    
    HYPERPARAMETER_SAMPLES = min(HYPERPARAMETER_SAMPLES, len(params))
    
    params = [
        params[_] for _ in np.random.choice(
            len(params),
            HYPERPARAMETER_SAMPLES,
            replace=False
            )
        ]

    for fold in tqdm(range(args.cv_folds)):

        model = {
            'dsm':DeepSurvivalMachines,
            'cph':DeepCoxPH,
            'dcm':DeepCoxMixtures,
            'cmhe':DeepCoxMixturesHeterogenousEffects
                 }[args.model_name]

        x = features[folds!=fold]
        t = outcomes.time[folds!=fold]
        e = outcomes.event[folds!=fold]

        x_tr, x_val = x[:tr_size], x[tr_size:]
        t_tr, t_val = t[:tr_size], t[tr_size:]
        e_tr, e_val = e[:tr_size], e[tr_size:]

        x_te = features[folds==fold]
        t_te = outcomes.time[folds==fold]
        e_te = outcomes.event[folds==fold]

        model_dict = {}
        for param in params:

            param['layers'] = param['nodes_'] * param['layers_']

            model_ = model(**param
                )

            model_, loss = model_.fit(
                x=x_tr.values,
                t=t_tr.values,
                e=e_tr.values,
                val_data=(
                    x_val.values,
                    t_val.values,
                    e_val.values
                    ),
                iters=args.epochs,
                **param
                )

            model_dict[model_] = np.mean(loss)

        model_ = min(model_dict, key=model_dict.get)

        try:
            out_risk = model_.predict_risk(x_te.values, times)
            out_risk = np.nan_to_num(out_risk, 1)
            out_survival = 1 - out_risk
        except:
            out_survival = model_.predict_survival(x_te.values, times)
            out_survival = np.nan_to_num(out_survival, 0)
            out_risk = 1 - out_survival

        cis = []
        brs = []

        et_tr = np.array(
            [(e_tr.values[i], t_tr.values[i]) for i in range(len(e_tr))],
                          dtype = [('e', bool), ('t', float)])
        et_te = np.array(
            [(e_te.values[i], t_te.values[i]) for i in range(len(e_te))],
                          dtype = [('e', bool), ('t', float)])
        et_val = np.array(
            [(e_val.values[i], t_val.values[i]) for i in range(len(e_val))],
                          dtype = [('e', bool), ('t', float)])

        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(
                    et_tr,
                    et_te,
                    out_risk[:, i],
                    times[i]
                    )[0]
                )

        max_te = max([k[1] for k in et_te])
        max_tr = max([k[1] for k in et_tr])

        while max_te > max_tr:
            idx = [k[1] for k in et_te].index(max_te)
            et_te = np.delete(et_te, idx, 0)
            out_survival = np.delete(out_survival, idx, 0)
            out_risk = np.delete(out_risk, idx, 0)
            max_te = max([k[1] for k in et_te])

        brs.append(brier_score(et_tr, et_te, out_survival, times)[1])
        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(
                cumulative_dynamic_auc(
                    et_tr,
                    et_te,
                    out_risk[:, i],
                    times[i]
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

    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
            ]

    os.makedirs('./fold_results', exist_ok=True)
    fold_results.to_csv(
        './fold_results/fold_results_{}_{}.csv'.format(
            args.dataset,
            args.model_name
            )
        )

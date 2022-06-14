# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:14:40 2022

@author: Mert
"""

import argparse
from collections import defaultdict

import numpy as np

from sklearn.model_selection import ParameterGrid
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from tqdm import tqdm

from datasets import load_dataset
from auton_lab.auton_survival.models.dsm import DeepSurvivalMachines
from auton_lab.auton_survival.models.cph import DeepCoxPH
from auton_lab.auton_survival.models.dcm import DeepCoxMixtures
from auton_lab.auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--model', default='dsm', type=str)
    parser.add_argument('--dataset', default='flchain', type=str)

    args = parser.parse_args()
    
    model = {
        'dsm':DeepSurvivalMachines,
        'cph':DeepCoxPH,
        'dcm':DeepCoxMixtures,
        'cmhe':DeepCoxMixturesHeterogenousEffects
             }[args.model]
    
    param_grid = {'dsm': {'k' : [3, 4, 6],
                          'distribution' : ['LogNormal', 'Weibull'],
                          'learning_rate' : [ 1e-4, 1e-3],
                          'layers' : [ [50], [50, 50], [100], [100, 100] ],
                          'discount': [ 1/2, 3/4, 1 ]},
                  'cph': {'layers' : [ [50], [50, 50], [100], [100, 100] ]},
                  'dcm': {'k' : [3, 4, 6], 
                          'layers' : [ [50], [50, 50], [100], [100, 100] ],
                          'batch_size': [ 128 ]},
                  'cmhe':{}
                  }[args.model]

    outcomes, features = load_dataset(args.dataset)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()

    unique_times = np.unique(outcomes['time'].values)

    n = len(features)
    tr_size = int(n*0.7)

    folds = np.array(list(range(args.cv_folds))*n)[:n]
    np.random.shuffle(folds)

    fold_results = defaultdict(lambda: defaultdict(list))

    for fold in tqdm(range(args.cv_folds)):

        x = features[folds!=fold]
        t = outcomes.time[folds!=fold]
        e = outcomes.event[folds!=fold]

        x_train, x_val = x[:tr_size], x[tr_size:]
        t_train, t_val = t[:tr_size], t[tr_size:]
        e_train, e_val = e[:tr_size], e[tr_size:]

        x_test = features[folds==fold]
        t_test = outcomes.time[folds==fold]
        e_test = outcomes.event[folds==fold]

        params = ParameterGrid(param_grid)

        model_dict = {}
        for param in params:
            model_ = model(**param
                )

            model_, loss = model_.fit(
                x=x_train.values,
                t=t_train.values,
                e=e_train.values,
                val_data=(
                    x_val.values,
                    t_val.values,
                    e_val.values
                    ),
                iters=1000,
                **param
                )

            model_dict[model_] = np.mean(loss)

        model_ = min(model_dict, key=model_dict.get)
        
        try:
            out_risk = model_.predict_risk(x_test.values, times)
            out_survival = 1 - out_risk
        except:
            out_survival = model_.predict_survival(x_test.values, times)
            out_risk = 1 - out_survival
            
        cis = []
        brs = []

        et_train = np.array(
            [(e_train.values[i], t_train.values[i]) for i in range(len(e_train))],
                          dtype = [('e', bool), ('t', float)])
        et_test = np.array(
            [(e_test.values[i], t_test.values[i]) for i in range(len(e_test))],
                          dtype = [('e', bool), ('t', float)])
        et_val = np.array(
            [(e_val.values[i], t_val.values[i]) for i in range(len(e_val))],
                          dtype = [('e', bool), ('t', float)])

        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(
                    et_train, 
                    et_test, 
                    out_risk[:, i], 
                    times[i]
                    )[0]
                )

        brs.append(brier_score(et_train, et_test, out_survival, times)[1])
        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(
                cumulative_dynamic_auc(
                    et_train, 
                    et_test, 
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

# -*- coding: utf-8 -*-
"""
Created on Sat May 28 19:14:40 2022

@author: Mert
"""

import argparse
import numpy as np

from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
    )
from sklearn.model_selection import ParameterGrid

import torch
from tqdm import tqdm

from auton_survival import datasets, preprocessing
from auton_survival.estimators import SurvivalModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--random_seed', default=1, type=int)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--model', default='dsm', type=str)
    args = parser.parse_args()

    outcomes, features = datasets.load_dataset("SUPPORT")

    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = [key for key in features.keys() if key not in cat_feats]

    features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
        )

    n = len(features)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()

    unique_times = np.unique(outcomes['time'].values)

    folds = np.array(list(range(args.cv_folds))*n)[:n]
    np.random.shuffle(folds)
            
    hyperparam_grid = {'k' : [3, 4, 6],
                  'distribution' : ['LogNormal', 'Weibull'],
                  'learning_rate' : [ 1e-4, 1e-3],
                  'layers' : [ [50], [50, 50], [100], [100, 100] ],
                  'discount': [ 1/2, 3/4, 1 ]
                 }

    hyperparam_grid = list(ParameterGrid(hyperparam_grid))
    
    for hyper_param in tqdm(hyperparam_grid):
    
      predictions = np.zeros((len(features), len(times)))
    
      fold_models = {}
      for fold in tqdm(range(args.cv_folds)):
        # Fit the model
        fold_model = SurvivalModel(
            model=args.model, 
            random_seed=args.random_seed,
            **hyper_param
            )    
        fold_model.fit(features.loc[folds!=fold], outcomes.loc[folds!=fold])
        fold_models[fold] = fold_model

        # Predict risk scores
        predictions[folds==fold] = fold_model.predict_survival(
            features.loc[folds==fold], 
            times=times
            )
        # Evaluate IBS
      score_per_fold = []
      for fold in range(args.cv_folds):
          
          out_survival = predictions[folds==fold]
          out_risk = 1 - out_survival
          f_tr, o_tr = features.loc[folds!=fold], outcomes.loc[folds!=fold]
          
          f_te, o_te = features.loc[folds==fold], outcomes.loc[folds==fold]
          
          x_train, t_train, e_train = f_tr, o_tr.time, o_tr.event
          x_test, t_test, e_test = f_te, o_te.time, o_te.event

          et_train = np.array(
            [(e_train.values[i], t_train.values[i]) for i in range(len(e_train))],
                         dtype = [('e', bool), ('t', float)])

          et_test = np.array(
            [(e_test.values[i], t_test.values[i]) for i in range(len(e_test))],
                         dtype = [('e', bool), ('t', float)])
          
          cis = []
          brs = []
  
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



    #     score_per_fold.append(score)

    #   current_score = np.mean(score_per_fold)

    #   if current_score < best_score:
    #     best_score = current_score
    #     best_model = fold_models
    #     best_hyper_param = hyper_param
    #     best_predictions = predictions

    # self.best_hyperparameter = best_hyper_param
    # self.best_model_per_fold = best_model
    # self.best_predictions = best_predictions

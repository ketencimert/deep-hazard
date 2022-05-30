# -*- coding: utf-8 -*-
"""
Created on Fri May 27 18:27:31 2022

@author: Mert
"""

import argparse

from copy import deepcopy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from itertools import chain

from auton_survival import datasets, preprocessing
from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
    )


outcomes, features = datasets.load_dataset("SUPPORT")

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = [key for key in features.keys() if key not in cat_feats]

features = preprocessing.Preprocessor().fit_transform(
    cat_feats=cat_feats,
    num_feats=num_feats,
    data=features,
    )

horizons = [0.25, 0.5, 0.75]
times = np.quantile(outcomes.time[outcomes.event==1], horizons).tolist()
cv_folds = 5
n = len(features)
folds = np.array(list(range(cv_folds))*n)[:n]
np.random.shuffle(folds)


unique_times = np.unique(outcomes['time'].values)

time_min, time_max = unique_times.min(), unique_times.max()

for fold in range(self.cv_folds):

    fold_outcomes = outcomes.loc[folds==fold, 'time']
    
    if fold_outcomes.min() > time_min: time_min = fold_outcomes.min()
    if fold_outcomes.max() < time_max: time_max = fold_outcomes.max()
  
unique_times = unique_times[unique_times>=time_min]
unique_times = unique_times[unique_times<time_max]

scores = []

best_model = {}
best_score = np.inf

self.hyperparam_grid = list(ParameterGrid(hyperparam_grid))

for hyper_param in tqdm(self.hyperparam_grid):

    predictions = np.zeros((len(features), len(unique_times)))
    
    fold_models = {}
    for fold in tqdm(range(self.cv_folds)):
        # Fit the model
        fold_model = SurvivalModel(model=self.model, random_seed=self.random_seed, **hyper_param)    
        fold_model.fit(features.loc[folds!=fold], outcomes.loc[folds!=fold])
        fold_models[fold] = fold_model
    
        # Predict risk scores
        predictions[folds==fold] = fold_model.predict_survival(features.loc[folds==fold], times=unique_times)
        # Evaluate IBS
    score_per_fold = []
    for fold in range(self.cv_folds):
        score = survival_regression_metric('ibs', predictions, outcomes, unique_times, folds, fold)
        score_per_fold.append(score)
    
    current_score = np.mean(score_per_fold)
    
    if current_score < best_score:
      best_score = current_score
      best_model = fold_models
      best_hyper_param = hyper_param
      best_predictions = predictions

self.best_hyperparameter = best_hyper_param
self.best_model_per_fold = best_model
self.best_predictions = best_predictions

if ret_trained_model:

  model = SurvivalModel(model=self.model, random_seed=self.random_seed, **self.best_hyperparameter)
  model.fit(features, outcomes)


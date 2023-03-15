# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:07:48 2023

@author: Mert
"""
import ast
import os
import operator
import glob, os
import itertools
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

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
sns.set_palette("Dark2")

avg_fold_results = defaultdict(lambda: defaultdict(list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Fixed parameters
    parser.add_argument('--dataset', default='metabric', type=str)
    args = parser.parse_args()
    
    avg_fold_results = defaultdict(lambda: defaultdict(list))
    
    for file in os.listdir('./fold_results'):
        if file.endswith(".csv"):
            if args.dataset in file:
                fold_results = pd.read_csv(
                    os.path.join('./fold_results', file),
                    index_col=0
                    )

                for key1 in fold_results.index:
                    for key2 in fold_results.loc[key1].keys():

                        avg_fold_results[key1][int(
                            key2.split(' ')[-1]
                            )].append(
                            fold_results.loc[key1][key2]
                            )

    for key1 in avg_fold_results.keys():
        for key2 in avg_fold_results[key1].keys():
            avg_fold_results[key1][key2] = np.mean(
                avg_fold_results[key1][key2]) 

    avg_fold_results = pd.DataFrame(avg_fold_results).T
    fig, axes = plt.subplots(4, 3, figsize=(20,15))
    
    fig.suptitle('{} Dataset\n'.format(
        args.dataset[0].upper() + args.dataset[1:]
        ),
        fontsize=30
        )

    s = 4
    e = 300
    k = range(0, 4)
    j = range(0, 3)
    idx = []
    for k_ in k:
        for j_ in j:
            idx.append((k_, j_))
    
    keys = list(
        avg_fold_results.index[:-3]
        ) + [
            avg_fold_results.index[-2]
            ] + [
                avg_fold_results.index[-3]
                ] + [avg_fold_results.index[-1]]
    
    for i, key in enumerate(keys):
        k, j = idx[i]
        
        axes[k][j].set_title(key, fontsize=22)
        axes[k][j].tick_params(labelsize=15)

        x = avg_fold_results.loc[key].index
        y = avg_fold_results.loc[key].values
        x, y = [np.asarray(l) for l in zip(*sorted(zip(x, y)))]
        x, y = x[(s<x) * (x<e)], y[(s<x) * (x<e)]
        
        
        sns.regplot(
        x, y, color='black',ax=axes[k][j]
         )

        y_maxes = y.argmax(0)
        axes[k][j].scatter(
        x[y_maxes],
        y[y_maxes],
        color='blue' if ('C-Index' in key) or ('AUC' in key) else 'red',
        )
        axes[k][j].axvline(
            x=x[y_maxes], linestyle='--',
            color='blue' if ('C-Index' in key) or ('AUC' in key) else 'red',
            label='Best' if ('C-Index' in key) or ('AUC' in key) else 'Worst'
            )
        # axes[k][j].axeshline(y=y[y_maxes], color='black', linestyle='--')

        y_min = y.argmin(0)
        axes[k][j].scatter(
        x[y_min],
        y[y_min],
        color='red' if ('C-Index' in key) or ('AUC' in key) else 'blue',
        )
        
        axes[k][j].axvline(
            x=x[y_min], linestyle='--',
            color='red' if ('C-Index' in key) or ('AUC' in key) else 'blue',
            label='Worst' if ('C-Index' in key) or ('AUC' in key) else 'Best'
            )
        # axes[k][j].plot(
        #     [x[y_min]+15, x[y_min]+15], [y[y_min], y[y_maxes]],
        #     color = 'black')
        # axes[k][j].plot(
        #     [x[y_min], x[y_min]+15], [y[y_min], y[y_min]],
        #     color = 'black')
        # axes[k][j].plot(
        #     [x[y_min], x[y_min]+15], [y[y_maxes], y[y_maxes]],
        #     color = 'black')
        axes[k][j].legend(fontsize=15)

        # axes[k][j].axeshline(y=y[y_min], color='black', linestyle='--')

        # axes[k][j].annotate('t-s', xy=(0.725, -0.15), xytext=(0.725, -0.35),
        #     fontsize=14, ha='center', va='bottom', xycoords='axeses fraction', 
        #     bbox=dict(boxstyle='square', fc='0.8'),
        #     arrowprops=dict(arrowstyle='-[, widthB=5.0, lengthB=.5', lw=2.0))
    plt.tight_layout()
    plt.show()
        
        

    
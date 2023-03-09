# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:56:58 2022

@author: Mert
"""
import random
import os
from collections import defaultdict
import numpy as np

import argparse
import matplotlib.pyplot as plt

import torch
from torch.distributions.uniform import Uniform
from torch.utils.data import DataLoader

from utils import get_survival_curve, get_hazard_curve
from datasets import SurvivalData, load_dataset

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--bs', default=1024, type=str)

    parser.add_argument('--dtype', default='float64', type=str)
    parser.add_argument('--dataset', default='metabric', type=str)
    parser.add_argument('--importance_samples', default=256, type=int)
    parser.add_argument('--sample', default=6001, type=int)
    args = parser.parse_args()

    outcomes, features = load_dataset(args.dataset)
    dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[args.dtype]
    SEED = 12345
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    x, t, e = features, outcomes.time, outcomes.event
    n = len(features)
    tr_size = int(n * 0.7)
    all_times = [t.min(), t.max()]
    folds = np.array(list(range(5)) * n)[:n]
    np.random.shuffle(folds)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e == 1], horizons).tolist()

    fold_results = defaultdict(lambda: defaultdict(list))

    lambdann = torch.load(
        './model_checkpoint/{}.pth'.format(args.dataset),
        map_location='cuda:0'
        ).to(args.device)
    fold = 1
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
        x_te.values[:300], t_te.values[:300], e_te.values[:300], args.device, dtype
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
    hazards = get_hazard_curve(
        lambdann, 
        test_dataloader, 
        [0, t.max()], 
        20,
        slices=0.05
        )
    patient1 = 173
    patient2 = 263
    patient3 = 246
    size = 15
    x = hazards.index
    y1 = hazards.values.T[patient1]
    y2 = hazards.values.T[patient2]
    y3 = hazards.values.T[patient3]
    
    c1 = 'blue'
    c2 = 'red'
    c3 = 'green'
    
    ax = plt.subplot(111)
    ax.plot(x, y1, lw=2, color=c1, linestyle='dashed', label = 'Instance 1')
    ax.plot(x, y2, lw=2, color=c2, linestyle='dashed', label = 'Instance 2')
    ax.plot(x, y3, lw=2, color=c3, linestyle='dashed', label = 'Instance 3')
    ax.legend(prop={'size': size}, loc='upper right')
    ax.fill_between(x, 0, y1, alpha=0.03, color=c1)
    ax.fill_between(x, 0, y2, alpha=0.03, color=c2)
    ax.fill_between(x, 0, y3, alpha=0.03, color=c3)
    ax.set_xlabel('Time',size=25)
    ax.set_ylabel('Hazard Rate',size=25, labelpad=10)
    # majorLocator = MultipleLocator(1)
    # ax.xaxis.set_major_locator(majorLocator)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.set_ylabel('Hazard Rate', fontsize=size)
    # ax.set_xlabel('Time', fontsize=size)
    ax.tick_params(axis='both', which='major', labelsize=size)
    # ax.set_title('PBC Dataset', fontsize=size)
    plt.tight_layout()
    plt.savefig("./hazard_figures/{}_hazard.svg".format(
            args.dataset,
            )
        )
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:56:58 2022

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--dtype', default='float64', type=str)
    parser.add_argument('--importance_samples', default=256, type=int)
    parser.add_argument('--sample', default=0, type=int)
    args = parser.parse_args()

    dtype = {
        'float64':torch.double,
        'float32':torch.float,
        }[args.dtype]

    outcomes, features = datasets.load_dataset("SUPPORT")

    cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
    num_feats = [key for key in features.keys() if key not in cat_feats]

    features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
        )

    x, t, e = features, outcomes.time, outcomes.event
    
    with torch.no_grad():
    
        x = torch.tensor(x.values[args.sample], dtype=dtype).to(args.device)   
        time_line = range(1, t.max())
        
        best_lambdann = torch.load('./best_lambdann.pth').eval()
    
        density = []
    
        for time in time_line:
            
            t = torch.tensor([time], dtype=dtype).to(args.device)
            t = t.view(1,1)
            x = x.view(1,-1)
    
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample((args.importance_samples,)).T
    
            density.append(
                torch.exp(
                    best_lambdann(x=x, t=t).log()
                    - torch.mean(
                        best_lambdann(x=x, t=t_samples).view(x.size(0), -1),
                        -1) * t
                    ).squeeze(-1)
                )
        
        density = torch.cat(density)
    
        plt.plot(density.detach().cpu().numpy(), color='b')
        
    
    
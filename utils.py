# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:06:30 2022

@author: Mert
"""

import numpy as np

import torch
from torch.distributions.uniform import Uniform

import pandas as pd
from pycox.evaluation import EvalSurv

from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
)

def build_times(times, slices):
    times_ = []
    for i in range(len(times)):
        if i != 0:
            t1 = int(times[i-1])
        else:
            t1 = 0
        insert = list(
            range(t1, int(np.floor(times[i])), slices)
            )
        insert.pop(0)
        times_.append(insert)
        times_.append([times[i]])
    times_.insert(0,[0])
    times_ = [float(item) for sublist in times_ for item in sublist]
    
    indexes = [times_.index(q) for q in times]
    
    return times_, indexes

def compute_survival(model, x, times, imps, slices=1, return_all=False):
    """
    computes the survival curve for one instance using DP returns 1xsurvival time
    """
    with torch.no_grad():
        dtype = model.feature_net[0].weight.dtype
        quantiles_, indexes = build_times(times, slices)
        survival = [
            torch.ones(x.shape[0], dtype=dtype)
            ]
        for i in range(len(quantiles_)-1):
            t1 = quantiles_[i]
            t2 = quantiles_[i+1]
            importance_sampler = Uniform(t1, t2)
            t_samples_ = importance_sampler.sample(
                (x.shape[0], imps)
                ).to(x.device).unsqueeze(-1).type(dtype)
            int_lambdann = torch.mean(model(
                x=x,
                t=t_samples_
                ).view(x.size(0), -1), -1) * (t2 - t1)
            survival.append(
                torch.exp(survival[-1].log() -  int_lambdann.cpu())
                )
        if return_all:
            survival = torch.stack(survival, -1)[:,indexes[0]: indexes[1]]
        else:
            survival = torch.stack(survival, -1)[:,indexes]
        return survival

def evaluate_model(
        model, batcher, times, train, valid, 
        bs, imps, return_all=False
        ):

    with torch.no_grad():
        loglikelihoods = []
        survival = []
        for (x, t, e) in batcher:
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample(
                (imps,)
            ).T

            loglikelihood = (
                    model(x=x, t=t).log().squeeze(-1) * e
                    - torch.mean(
                model(x=x, t=t_samples).view(x.size(0), -1),
                -1) * t
            ).mean()

            loglikelihoods.append(loglikelihood.item())

            # For C-Index and Brier Score
            survival.append(
                compute_survival(
                    model, 
                    x,
                    times,
                    imps, 
                    slices=1,
                    return_all=return_all
                    )
                )
        survival = torch.cat(survival).numpy()
        if return_all:
            times_, indexes = build_times(times, slices=1)
            survival = pd.DataFrame(
                survival.T, index=times_[indexes[0]:indexes[1]]
                )
            t = np.asarray([t[1] for t in valid])
            e = np.asarray([t[0] for t in valid])
            ev = EvalSurv(survival, t, e, censor_surv='km')
            cis = ev.concordance_td('antolini')
            brs = ev.brier_score(
                np.linspace(times[0], times[1], 100)
                ).mean()
            roc_auc = None
        else:
            risk = 1 - survival
            cis = []
            brs = []
            for i, _ in enumerate(times):
                cis.append(
                    concordance_index_ipcw(
                        train, valid, risk[:, i], times[i]
                    )[0]
                )

            max_val = max([k[1] for k in valid])
            max_tr = max([k[1] for k in train])
            while max_val > max_tr:
                idx = [k[1] for k in valid].index(max_val)
                valid = np.delete(valid, idx, 0)
                survival = np.delete(survival, idx, 0)
                risk = np.delete(risk, idx, 0)
                max_val = max([k[1] for k in valid])
    
            brs.append(
                brier_score(
                    train, valid, survival, times
                )[1]
            )
    
            roc_auc = []
            for i, _ in enumerate(times):
                roc_auc.append(
                    cumulative_dynamic_auc(
                        train, valid, risk[:, i], times[i]
                    )[0]
                )
            
        return np.mean(loglikelihoods), cis, brs, roc_auc, survival
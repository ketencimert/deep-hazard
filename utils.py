# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:06:30 2022

@author: Mert
"""

import numpy as np

import pandas as pd
import torch
from torch.distributions.uniform import Uniform
from tqdm import tqdm

from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
)

def build_times(times, slices):
    times_ = [times[0]]
    for i in range(int(np.ceil(times[0])), int(np.floor(times[1])), 1):
        if i != times_[0]:
            times_.append(i)
    if times_[-1] != times[1]:
        times_.append(times[1])
    indexes = [times_.index(q) for q in times]
    return times_, indexes

def compute_survival(model, x, times, imps, slices=1):
    """
    computes the survival curve for one instance using DP returns 1xsurvival time
    """
    with torch.no_grad():
        dtype = model.feature_net[0].weight.dtype
        quantiles_, indexes = build_times(
            times,
            slices
            )
        survival = [
            torch.ones(x.shape[0], dtype=dtype)
            ]
        for i in range(len(quantiles_) - 1):
            t1 = quantiles_[i]
            t2 = quantiles_[i+1]
            importance_sampler = Uniform(t1, t2)
            t_samples_ = importance_sampler.sample(
                (x.shape[0], imps//10)
                ).to(x.device).unsqueeze(-1).type(dtype)
            int_lambdann = torch.mean(
                model(
                x=x,
                t=t_samples_
                ), -1) * (t2 - t1)
            survival.append(
                torch.exp(survival[-1].log() -  int_lambdann.cpu())
                )
        survival = torch.stack(survival, -1)[:,indexes[0]: indexes[1]+1]
        return survival

def get_survival_curve(model, batcher, times, imps):
    with torch.no_grad():
        survival = []
        for (x, t, e) in tqdm(
                batcher, total=batcher.dataset.__blen__()
                ):
            survival.append(
                compute_survival(
                    model,
                    x,
                    times,
                    imps,
                    slices=1,
                    )
                )
        survival = torch.cat(survival).numpy()
        times_, indexes = build_times(times, slices=1)
        survival = pd.DataFrame(
            survival.T, index=times_[indexes[0]:indexes[1]+1]
            )
        return survival

def evaluate_model(model, batcher, times, train, valid, bs, imps):

    dtype = model.feature_net[0].weight.dtype
    device = model.feature_net[0].weight.device

    with torch.no_grad():

        times_tensor = torch.tensor(times, dtype=dtype).to(device)
        times_tensor = times_tensor.unsqueeze(-1).repeat_interleave(
            bs, -1
        ).T

        importance_sampler = Uniform(0, times_tensor)
        t_samples_ = torch.transpose(
            importance_sampler.sample(
                (imps,)
            ), 0, 1
        )

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
                model(x=x, t=t_samples),
                -1) * t
            ).mean()

            loglikelihoods.append(loglikelihood.item())

            # For C-Index and Brier Score

            survival_quantile = []
            for i in range(len(times)):
                int_lambdann = torch.mean(
                    model(
                        x=x,
                        t=t_samples_[:x.size(0), :, i]),
                    -1) * times[i]

                survival_quantile.append(torch.exp(-int_lambdann))

            survival_quantile = torch.stack(survival_quantile, -1)
            survival.append(survival_quantile)

        survival = torch.cat(survival).cpu().numpy()
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

        return np.mean(loglikelihoods), cis, brs, roc_auc
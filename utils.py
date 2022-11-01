# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:06:30 2022

@author: Mert
"""

import numpy as np

import torch
from torch.distributions.uniform import Uniform

from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
)

def evaluate_model(model, batcher, quantiles, train, valid, 
                   bs, imps, dtype, device):

    with torch.no_grad():

        times_tensor = torch.tensor(quantiles, dtype=dtype).to(device)
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
                model(x=x, t=t_samples).view(x.size(0), -1),
                -1) * t
            ).mean()

            loglikelihoods.append(loglikelihood.item())

            # For C-Index and Brier Score

            survival_quantile = []
            for i in range(len(quantiles)):
                int_lambdann = torch.mean(
                    model(
                        x=x,
                        t=t_samples_[:x.size(0), :, i]).view(x.size(0), -1),
                    -1) * quantiles[i]

                survival_quantile.append(torch.exp(-int_lambdann))

            survival_quantile = torch.stack(survival_quantile, -1)
            survival.append(survival_quantile)

        survival = torch.cat(survival).cpu().numpy()
        risk = 1 - survival

        cis = []
        brs = []
        for i, _ in enumerate(quantiles):
            cis.append(
                concordance_index_ipcw(
                    train, valid, risk[:, i], quantiles[i]
                )[0]
            )

        #Remove larger test times to confirm with
        #https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
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
                train, valid, survival, quantiles
            )[1]
        )

        roc_auc = []
        for i, _ in enumerate(quantiles):
            roc_auc.append(
                cumulative_dynamic_auc(
                    train, valid, risk[:, i], quantiles[i]
                )[0]
            )

        return np.mean(loglikelihoods), cis, brs, roc_auc
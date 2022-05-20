# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:12:27 2022

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

from itertools import chain

from auton_survival import datasets, preprocessing
from sksurv.metrics import (
    concordance_index_ipcw, brier_score, cumulative_dynamic_auc
    )


def validate_model(model, batcher, times, et_tr, et_val):

    with torch.no_grad():
        times_tensor = torch.tensor(times).to(args.device).double()

        times_tensor = times_tensor.unsqueeze(-1).repeat_interleave(
            train_dataloader.batch_size,-1
            ).T

        importance_sampler = Uniform(0, times_tensor)
        t_samples_ = torch.transpose(
            importance_sampler.sample(
            (args.importance_samples,)
            ),0,1
            )

        loglikelihoods = []
        survival = []
        ts = []
        for (x, t, e) in batcher:
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample(
                (args.importance_samples,)
                ).T

            loglikelihood = (
                lambdann(x=x, t=t).log() * e
                - torch.mean(
                    lambdann(x=x, t=t_samples).view(x.size(0), -1),
                    -1)
                ).mean()

            loglikelihoods.append(loglikelihood.item())

            #For C-Index and Brier Score

            survival_quantile = []
            for i in range(len(times)):

                neg_logcdf = torch.mean(
                        lambdann(
                            x=x,
                            t=t_samples_[:x.size(0),:, i]).view(x.size(0), -1),
                        -1)

                survival_quantile.append(torch.exp(-neg_logcdf))

            survival_quantile = torch.stack(survival_quantile, -1)
            survival.append(survival_quantile)
            ts.append(t)

        ts = torch.cat(ts).cpu().numpy()
        survival = torch.cat(survival).cpu().numpy()
        risk = 1 - survival

        cis = []
        brs = []
        for i, _ in enumerate(times):
            cis.append(
                concordance_index_ipcw(
                    et_tr, et_val, risk[:, i], times[i]
                    )[0]
                )

        brs.append(
            brier_score(
                et_tr, et_val, survival, times
                )[1]
            )

        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(
                cumulative_dynamic_auc(
                    et_tr, et_val, risk[:, i], times[i]
                    )[0]
                )

        return np.mean(loglikelihoods), cis, brs, roc_auc


class SurvivalData(torch.utils.data.Dataset):
    def __init__(self, x, t, e, cuda, dtype=torch.double):

        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(t, dtype=dtype),
                torch.tensor(e, dtype=dtype)
                ] for x,t,e in zip(x,t,e)
            ]

        self.cuda = cuda
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.cuda:

                self._cache[index][0] = self._cache[
                    index][0].cuda(non_blocking=True)

                self._cache[index][1] = self._cache[
                    index][1].cuda(non_blocking=True)

                self._cache[index][2] = self._cache[
                    index][2].cuda(non_blocking=True)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_


class LambdaNN(nn.Module):
    def __init__(self, d_in, d_out, d_hid, n_layers, activation="relu",
                 p=0.3, dtype=torch.double):
        super().__init__()

        act_fn = nn.ReLU()
        self.feature_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                d_in if ii == 0 else d_hid,
                                d_in if ii + 1 == n_layers else d_hid,
                                dtype=dtype
                            ),
                            nn.Identity() if ii + 1 == n_layers else act_fn,
                            nn.Dropout(p)
                        ]
                        for ii in range(n_layers)
                    ]
                )
            )
        self.feature_net.pop(-1)
        self.feature_net = nn.Sequential(*self.feature_net)

        self.time_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                1 if ii == 0 else d_hid,
                                d_in if ii + 1 == n_layers else d_hid,
                                dtype=dtype
                            ),
                            nn.Identity() if ii + 1 == n_layers else act_fn,
                            nn.Dropout(p)
                        ]
                        for ii in range(n_layers)
                    ]
                )
            )
        self.time_net.pop(-1)
        self.time_net = nn.Sequential(*self.time_net)

        self.shared_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                int(2*d_in) if ii == 0 else d_hid,
                                d_out if ii + 1 == n_layers else d_hid,
                                dtype=dtype
                            ),
                            nn.Identity() if ii + 1 == n_layers else act_fn,
                            nn.Dropout(p)
                        ]
                        for ii in range(n_layers)
                    ]
                )
            )
        self.shared_net.pop(-1)
        self.shared_net = nn.Sequential(*self.shared_net)

    def forward(self, x, t):

        x = self.feature_net(x)
        t = self.time_net(t.reshape(-1,1))

        if x.size(0) != t.size(0):

            x = x.repeat_interleave(t.size(0) // x.size(0), 0)

        z = self.shared_net(torch.cat([x, t], -1))

        return nn.Softplus()(z)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument('--device', default='cuda', type=str)
    #optimization args
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--importance_samples', default=100, type=int)
    #model, encoder-decoder args
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    args = parser.parse_args()

    outcomes, features = datasets.load_dataset("SUPPORT")

    cat_feats = [
        'sex',
        'dzgroup',
        'dzclass',
        'income',
        'race',
        'ca'
        ]

    num_feats = [key for key in features.keys() if key not in cat_feats]

    features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
        )

    x, t, e = features, outcomes.time, outcomes.event
    n = len(x)

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e==1], horizons).tolist()

    tr_size = int(n*0.70)
    vl_size = int(n*0.10)
    te_size = int(n*0.20)

    x_tr, x_te, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
    t_tr, t_te, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
    e_tr, e_te, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]

    et_tr = np.array(
        [
            (e_tr.values[i], t_tr.values[i]) for i in range(len(e_tr))
            ],
                     dtype = [('e', bool), ('t', float)]
                     )

    et_te = np.array(
        [
            (e_te.values[i], t_te.values[i]) for i in range(len(e_te))
            ],
                     dtype = [('e', bool), ('t', float)]
                     )

    et_val = np.array(
        [
            (e_val.values[i], t_val.values[i]) for i in range(len(e_val))
            ],
                     dtype = [('e', bool), ('t', float)]
                     )

    d_in = x_tr.shape[1]
    d_out = 1
    d_hid = d_in//2

    lambdann =  LambdaNN(
        d_in, d_out, d_hid, args.n_layers, p=args.dropout
        ).to(args.device)

    train_data = SurvivalData(
        x_tr.values, t_tr.values, e_tr.values, args.device
        )
    valid_data = SurvivalData(
        x_val.values, t_val.values, e_val.values, args.device
        )
    test_data = SurvivalData(
        x_te.values, t_te.values, e_te.values, args.device
        )

    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
        )
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False
        )
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
        )

    optimizer = optim.Adam(lambdann.parameters(), lr=args.lr)
    
    epoch_losses = defaultdict(list)
    
    epoch_tr_loglikelihoods = []
    epoch_val_loglikelihoods = []
    epoch_c_idxes = []

    tr_loglikelihood = - np.inf
    val_loglikelihood = - np.inf
    c_idx = - np.inf

    for epoch in range(args.epochs):

        print("Epoch: {}, LL_train: {}, LL_valid: {}".format(
            epoch, tr_loglikelihood, val_loglikelihood)
            )
        
        lambdann.train()
        tr_loglikelihoods = []
        for (x, t, e) in train_dataloader:

            optimizer.zero_grad()

            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample((args.importance_samples,)).T

            train_loglikelihood = (
                lambdann(x=x, t=t).log() * e
                - torch.mean(
                    lambdann(x=x, t=t_samples).view(x.size(0), -1),
                    -1) * t
                ).mean()

            tr_loglikelihoods.append(train_loglikelihood.item())
            
            #minimize negative loglikelihood
            (-train_loglikelihood).backward()
            optimizer.step()
                
        tr_loglikelihood = np.mean(tr_loglikelihoods)
        
        print("\nValidating Model...")
        #validate the model
        val_loglikelihood, cis, brs, roc_auc = validate_model(
            lambdann.eval(), valid_dataloader, times, et_tr, et_val
            )

        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index:", cis[horizon[0]])
            print("Brier Score:", brs[0][horizon[0]])
            print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

        epoch_losses['LL_train'].append(tr_loglikelihood)
        epoch_losses['LL_valid'].append(val_loglikelihood)
        epoch_losses['C-Index 0.25 quantile'].append(cis[0])
        epoch_losses['C-Index 0.5 quantile'].append(cis[1])
        epoch_losses['C-Index 0.75 quantile'].append(cis[2])

        if epoch_losses['LL_valid'][-1] == max(epoch_losses['LL_valid']):
            best_lambdann = deepcopy(lambdann)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

    ax[0].plot(epoch_losses['LL_train'], color='b', label="LL_train")
    ax_twin = ax[0].twinx()
    ax_twin.plot(epoch_losses['LL_valid'], color='r', label="LL_valid")
    ax[0].legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    color = ['r', 'g', 'b']
    i = 0
    for (key, value) in epoch_losses.items():
        if 'C-Index' in key:
            ax[1].plot(value, color=color[i], label=key)
            ax[1].legend(loc="upper left")
            i += 1
            
print("\nEvaluating Best Model...")
test_loglikelihood, cis, brs, roc_auc = validate_model(
            lambdann.eval(), test_dataloader, times, et_tr, et_te
            )
print("Test Loglikelihood: {}".format(test_loglikelihood))
for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
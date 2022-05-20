# -*- coding: utf-8 -*-
"""
Created on Thu May 19 23:12:27 2022

@author: Mert
"""
import argparse

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions.uniform import Uniform

from itertools import chain

from auton_survival import datasets, preprocessing
from lifelines.utils import concordance_index


def validate_model(model, batcher):
    
    with torch.no_grad():
        loglikelihoods = []
        cdfs = []
        ts = []
        for (x, t, e) in batcher:
            importance_sampler = Uniform(0, t)
            t_samples = importance_sampler.sample(
                (args.importance_samples*10,)
                ).T
            
            neg_logcdf = torch.mean(
                    lambdann(x=x, t=t_samples).view(x.size(0), -1),
                    -1)
            
            #this is an approximation to cdf - will neverbe exactly equal
            cdfs.append(1 - neg_logcdf.exp())
            ts.append(t)
            
            loglikelihood = (
                lambdann(x=x, t=t).log() * e
                - neg_logcdf
                ).mean()
    
            loglikelihoods.append(loglikelihood.item())
        
        ts = torch.cat(ts)
        ts = ts.topk(ts.size(0))[1].cpu().numpy()
        
        cdfs = torch.cat(cdfs)
        cdfs = cdfs.topk(cdfs.size(0))[1].cpu().numpy()
        
        c_idx = concordance_index(ts, cdfs)
        
        return np.mean(loglikelihoods), c_idx


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

            if self.cuda:

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

    horizons = [0.25, 0.5, 0.75]
    times = np.quantile(t[e==1], horizons).tolist()

    n = len(x)

    tr_size = int(n*0.70)
    vl_size = int(n*0.10)
    te_size = int(n*0.20)

    x_tr, x_te, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size+vl_size]
    t_tr, t_te, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size+vl_size]
    e_tr, e_te, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size+vl_size]

    d_in = x_tr.shape[1]
    d_out = 1
    d_hid = d_in//2

    lambdann =  LambdaNN(
        d_in, d_out, d_hid, args.n_layers, p=args.dropout
        ).to(args.device)

    train_data = SurvivalData(x_tr.values, t_tr.values, e_tr.values, 'cuda')
    valid_data = SurvivalData(x_val.values, t_val.values, e_val.values, 'cuda')
    test_data = SurvivalData(x_te.values, t_te.values, e_te.values, 'cuda')

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

        #validate
        tr_loglikelihood = np.mean(tr_loglikelihoods)
        val_loglikelihood, c_idx = validate_model(
            lambdann.eval(), valid_dataloader
            )

        epoch_tr_loglikelihoods.append(tr_loglikelihood)
        epoch_val_loglikelihoods.append(val_loglikelihood)
        epoch_c_idxes.append(c_idx)
        
        if epoch_val_loglikelihoods[-1] == max(epoch_val_loglikelihoods):
            best_lambdann = deepcopy(lambdann)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))

    ax[0].plot(epoch_tr_loglikelihoods[3:], color='b', label="LL_train")
    ax_twin = ax[0].twinx()
    ax_twin.plot(epoch_val_loglikelihoods[3:], color='r', label="LL_valid")
    ax[0].legend(loc="upper left")
    ax_twin.legend(loc="upper right")
    ax[1].plot(epoch_c_idxes[3:], color='g', label="C-Index")
    ax[1].legend(loc="upper left")
    
print("Evaluating Best Model...")
test_loglikelihood, c_idx = validate_model(
    lambdann.eval(), test_dataloader
    )
print("Test Loglikelihood: {}, Test C-Index: {}".format(
    test_loglikelihood, c_idx
    )
    )
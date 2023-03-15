# -*- coding: utf-8 -*-
"""
Created on Sun May 29 20:22:26 2022

@author: Mert
"""

import torch
from torch import nn
from torch.distributions.normal import Normal

from itertools import chain

class LambdaNN(nn.Module):
    def __init__(self, d_in, d_out, d_hid, n_layers, activation="relu",
                 p=0.3, norm=False, dtype=torch.double, only_shared=False):
        super().__init__()

        act_fn = {
            'relu':nn.ReLU(),
            'elu':nn.ELU(),
            'selu':nn.SELU(),
            'silu':nn.SiLU()
            }

        act_fn = act_fn[activation]

        norm_fn = {
            'layer':nn.LayerNorm(d_hid, dtype=dtype),
            'batch':nn.BatchNorm1d(d_hid, dtype=dtype)
            }

        if norm in norm_fn.keys():
            norm_fn = norm_fn[norm]
        else:
            norm = False

        self.noise = nn.Dropout(p)
        if not only_shared:
            self.feature_net = list(
                    chain(
                        *[
                            [
                                nn.Linear(
                                    d_in if ii == 0 else d_hid,
                                    d_hid if ii + 1 == n_layers else d_hid,
                                    dtype=dtype
                                ),
                                nn.Identity() if ii + 1 == n_layers else act_fn,
                                nn.Identity() if not norm else norm_fn,
                                nn.Dropout(p)
                            ]
                            for ii in range(n_layers)
                        ]
                    )
                )
            self.feature_net.pop(-1)
            self.feature_net.pop(-1)
            self.feature_net = nn.Sequential(*self.feature_net)

            self.time_net = list(
                    chain(
                        *[
                            [
                                nn.Linear(
                                    1 if ii == 0 else d_hid,
                                    d_hid if ii + 1 == n_layers else d_hid,
                                    dtype=dtype
                                ),
                                nn.Identity() if ii + 1 == n_layers else act_fn,
                                nn.Identity() if not norm else norm_fn,
                                nn.Dropout(p)
                            ]
                            for ii in range(n_layers)
                        ]
                    )
                )
            self.time_net.pop(-1)
            self.time_net.pop(-1)
            self.time_net = nn.Sequential(*self.time_net)
            d_shared = int(2*d_hid)
        else:
            d_shared = d_in + 1

        self.only_shared = only_shared
        self.shared_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                d_shared if ii == 0 else d_hid,
                                d_out if ii + 1 == n_layers else d_hid,
                                dtype=dtype
                            ),
                            nn.Identity() if ii + 1 == n_layers else act_fn,
                            nn.Identity() if not norm else norm_fn,
                            nn.Dropout(p)
                        ]
                        for ii in range(n_layers)
                    ]
                )
            )
        self.shared_net.pop(-1)
        self.shared_net.pop(-1)
        self.shared_net = nn.Sequential(*self.shared_net)

    def forward(self, x, t):

        x = self.noise(x)
        
        if not self.only_shared:
            x = self.feature_net(x)

        if self.training:

            t = Normal(loc=t, scale=1).sample()
        
        if not self.only_shared:
            t = self.time_net(t.reshape(-1,1))
        else:
            t = t.reshape(-1,1)
        if x.size(0) != t.size(0):

            x = x.repeat_interleave(t.size(0) // x.size(0), 0)

        z = self.shared_net(torch.cat([x, t], -1))

        return nn.Softplus()(z)

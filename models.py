# -*- coding: utf-8 -*-
"""
Created on Sun May 29 20:22:26 2022

@author: Mert
"""

import torch
from torch import nn
from torch.distributions.normal import Normal

from itertools import chain

class DeepHazardMixture(nn.Module):
    def __init__(self, lambdanns, prior=None):
        super().__init__()
        self.lambdanns = nn.ModuleList(lambdanns)
        self.prior = prior

    def forward(self, c, x, t):
        return self.lambdanns[c](x, t)

    def forward_lambda(self, c, x, t):
        return self.lambdanns[c](x, t)

    def forward_prior(self, x):
        return self.prior(x)

    def mixture_size(self):
        return len(self.lambdanns)

class ExplainableLambdaNN(nn.Module):
    def __init__(self, d_in, d_out, d_hid, n_layers, activation="relu",
                 p=0.3, norm=False, dtype=torch.double):
        super().__init__()

        act_fn = {
            'relu':nn.ReLU(),
            'elu':nn.ELU(),
            'selu':nn.SELU(),
            'silu':nn.SiLU()
            }

        act_fn = act_fn[activation]

        self.noise = nn.Dropout(p)
        self.shared_net = []
        for _ in range(d_in):

            shared_net = list(
                    chain(
                        *[
                            [
                                nn.Linear(
                                    2 if ii == 0 else d_hid,
                                    1 if ii + 1 == n_layers else d_hid,
                                    dtype=dtype
                                ),
                                nn.Identity() if ii + 1 == n_layers else act_fn,
                            ]
                            for ii in range(n_layers)
                        ]
                    )
                )
            self.shared_net.append(nn.Sequential(*shared_net))

        self.shared_net = nn.ModuleList(self.shared_net)

    def forward(self, x, t):

        t = t.reshape(-1,1)

        if x.size(0) != t.size(0):

            x = x.repeat_interleave(t.size(0) // x.size(0), 0)

        if self.training:

            t = Normal(loc=t, scale=1).sample()

        z = nn.Softplus()(
            torch.cat(
                [
                    self.shared_net[i](
                        torch.cat(
                            [
                                x[:,i].view(-1,1), t
                                ],
                            -1)
                        ).unsqueeze(1) for i in range(x.size(1))
                    ],
                1)
            )

        z = torch.sum(z.squeeze(-1), -1)

        return z

class LambdaNN(nn.Module):
    def __init__(self, d_in, d_out, d_hid, n_layers, activation="relu",
                 p=0.3, norm=False, dtype=torch.double):
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

        self.shared_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                int(2*d_hid) if ii == 0 else d_hid,
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
        x = self.feature_net(x)

        if self.training:

            t = Normal(loc=t, scale=1).sample()

        t = self.time_net(t.reshape(-1,1))

        if x.size(0) != t.size(0):

            x = x.repeat_interleave(t.size(0) // x.size(0), 0)

        z = self.shared_net(torch.cat([x, t], -1))

        return nn.Softplus()(z)


class Prior(nn.Module):
    def __init__(self, d_in, mixture_size, d_hid, n_layers, activation="relu",
                 p=0.3, norm=False, dtype=torch.double):
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

        self.feature_net = list(
                chain(
                    *[
                        [
                            nn.Linear(
                                d_in if ii == 0 else d_hid,
                                mixture_size if ii + 1 == n_layers else d_hid,
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

    def forward(self, x):

        x = self.noise(x)
        x = self.feature_net(x)

        return nn.LogSoftmax(-1)(x)
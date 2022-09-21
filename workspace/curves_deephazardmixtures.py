# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:56:58 2022

@author: Mert
"""

import argparse
import matplotlib.pyplot as plt

import torch
from torch.distributions.uniform import Uniform

from auton_lab.auton_survival import datasets, preprocessing

from models import LambdaNN, DeepHazardMixture

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--dtype', default='float64', type=str)
    parser.add_argument('--importance_samples', default=256, type=int)
    parser.add_argument('--sample', default=6001, type=int)
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

    time_line = range(1, 2 * t.max() // 3)

    with torch.no_grad():

        best_deephazardmixture = torch.load(
            './saves/best_deephazardmixture.pth'
            ).eval()
        best_deephazardmixture = best_deephazardmixture.to(args.device)
        
        mixture_size = best_deephazardmixture.mixture_size()
        
        for i in range(0, args.sample):

            x_ = torch.tensor(x.values[i], dtype=dtype).to(args.device)

            density = []

            for time in time_line:

                t_ = torch.tensor([time], dtype=dtype).to(args.device)
                t_ = t_.view(1,1)
                x_ = x_.view(1,-1)

                importance_sampler = Uniform(0, t_)
                t_samples = importance_sampler.sample(
                    (args.importance_samples,)
                    ).T

                loglikelihood = [
                    best_deephazardmixture(c=j, x=x_, t=t_).log().squeeze(-1)
                    - torch.mean(
                        best_deephazardmixture(
                            c=j, 
                            x=x_, 
                            t=t_samples
                            ).view(x_.size(0), -1),
                        -1) * t_
                    for j in range(mixture_size)
                    ]

                loglikelihood = torch.stack(loglikelihood, -1)
                posterior = loglikelihood - loglikelihood.logsumexp(
                    -1
                    ).view(-1,1)
                posterior = posterior.exp()
                loglikelihood = torch.sum(
                loglikelihood.exp() * posterior, -1
                )
                density.append(loglikelihood)

            density = torch.cat(density)
            plt.title('Data Point: {}'.format(i))
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.plot(density.detach().cpu().numpy(), color='b')
            plt.show()

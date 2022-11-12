# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:17:53 2022

@author: Mert
"""
import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform 
from torch.distributions.exponential import Exponential
from torch.distributions.weibull import Weibull
from torch.distributions.bernoulli import Bernoulli

import pandas as pd

def SyntheticData(propotional=True):
    sample_size = 1000
    p_observed = 0.3
    
    if propotional:

        x1 = Uniform(-2, 2).sample((sample_size,))
        x2 = Uniform(-2, 2).sample((sample_size,))
    
        x = torch.stack([x1, x2], -1)
    
        g = np.log(5) * torch.exp(-(x1**2 + x2**2) / 2) 
    
        t1 = Exponential(rate=g).sample((1,)).view(-1)
        t2 = Exponential(rate=g).sample((1,)).view(-1)
        is_observed = Bernoulli(probs=p_observed).sample(t1.size()).view(-1)
        t = torch.cat([t1.view(-1,1), t2.view(-1,1)], -1)
        t = torch.squeeze(
            t1 * is_observed, -1) + torch.squeeze(
                torch.min(t, -1)[0] * ~is_observed.bool(), -1
                )
    
        data = torch.cat([x, t.view(-1,1)], -1)
        data = torch.cat([data, is_observed.view(-1,1)], -1)
        data = pd.DataFrame(data.numpy())
        data = data.rename(columns = {0:'x1', 1:'x2', 2:'time', 3:'event'})
        num_feats = ['x1', 'x2']
    else:
        x = Uniform(-2, 1).sample((sample_size,))
        lambda_ = 1/x**2
        scale = 1/lambda_
        k = 1.5 * torch.sigmoid(x+10)
        t1 = Weibull(scale=1/scale, concentration=k).sample((1,)).view(-1)
        t2 = Weibull(scale=1/scale, concentration=k).sample((1,)).view(-1)
        is_observed = Bernoulli(probs=p_observed).sample(t1.size()).view(-1)
        t = torch.cat([t1.view(-1,1), t2.view(-1,1)], -1)
        t = torch.squeeze(
            t1 * is_observed, -1) + torch.squeeze(
                torch.min(t, -1)[0] * ~is_observed.bool(), -1
                )

        data = torch.cat([x.view(-1,1), t.view(-1,1)], -1)
        data = torch.cat([data, is_observed.view(-1,1)], -1)
        data = pd.DataFrame(data.numpy())
        data = data.rename(columns = {0:'x', 1:'time', 2:'event'})
        num_feats = ['x']
    return data, num_feats
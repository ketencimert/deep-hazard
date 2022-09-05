# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 19:04:58 2022

@author: Mert
"""
import h5py    
import numpy as np
import pandas as pd
from datasets import load_dataset

def load_dataset_(dataset='support', 
                  filename="./support_train_test.h5", 
                  deepsurv=True,
                  scale_time=True
                  ):
    
    if not deepsurv:
        outcomes, features = load_dataset(dataset)
    else:
        if deepsurv:
            f1 = h5py.File(filename,'r+')
            e_tr, t_tr, x_tr = [
                np.asarray(f1['train'][key]) for key in f1['train'].keys()
                ]
            e_te, t_te, x_te = [
                np.asarray(f1['test'][key]) for key in f1['test'].keys()
                ]
            e = np.concatenate([e_tr, e_te])
            t = np.concatenate([t_tr, t_te])
            x = np.concatenate([x_tr, x_te])
            outcomes_ = dict()
            features_ = dict()
            
            feature_names = ['age',
            'sex', 
            'race', 
            'number of comorbidities', 
            'presence of diabetes', 
            'presence of dementia', 
            'presence of cancer',
            'mean arterial blood pressure', 
            'heart rate', 
            'respiration rate',
             'temperature', 
             'white blood cell count', 
             "serum’s sodium", 
             "serum’s creatinine"
             ]
            
            outcomes_['event'] = e
            outcomes_['time'] = t
            for i, key in enumerate(feature_names):
                features_[key] = x[:,i]
            outcomes = pd.DataFrame.from_dict(outcomes_)
            features = pd.DataFrame.from_dict(features_)
            # if standardize_features:
            #     features = (features - features.mean()) / features.std()
        else:
            outcomes, features = load_dataset(dataset)
        if scale_time:
            outcomes.time = (
                outcomes.time - outcomes.time.min()
                ) / (
                    outcomes.time.max() - outcomes.time.min()
                    ) + 1e-15
            
    return outcomes, features

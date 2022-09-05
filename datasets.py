# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:47:07 2022

@author: Mert
"""
import h5py
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sksurv.datasets import load_flchain
from tqdm import tqdm

from auton_lab.auton_survival import datasets, preprocessing


def one_hot_encode(dataframe, column):
    categorical = pd.get_dummies(dataframe[column], prefix=column)
    dataframe = dataframe.drop(column, axis=1)
    return pd.concat([dataframe, categorical], axis=1, sort=False)


def load_dataset(
        dataset='SUPPORT',
        path='C:/Users/Mert/Desktop/data.csv',
        scale_time=False
        ):

    if dataset.lower() == 'support':
        outcomes, features = datasets.load_dataset('SUPPORT')
        cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
        num_feats = [key for key in features.keys() if key not in cat_feats]
        features = preprocessing.Preprocessor().fit_transform(
            cat_feats=cat_feats,
            num_feats=num_feats,
            data=features,
            )

    elif dataset.lower() == 'aki/ckd':
        data = pd.read_csv(path, index_col=0)
        drop = ['event', 'time', 'cohort_end_date']
        features = [x for x in data.keys() if x not in drop]

        cat_feats = [x for x in features if x.split('.')[-1] == 'categorical']
        num_feats = [x for x in features if x not in cat_feats]

        total = len(cat_feats)
        print("One hot encoding categorical features...")
        for cat_feat in tqdm(cat_feats, total=total):
            data = one_hot_encode(data, cat_feat)

        for col_name in data.columns:
            try:
                data[col_name].fillna(data[col_name].median(), inplace=True)
            except:
                pass

        data = shuffle(data.dropna(axis=1))
        outcomes = data[['event', 'time']].reset_index(drop=True)
        features = data.drop(columns=drop).reset_index(drop=True)
        features = (features-features.mean())/(features.std() + 1e-5)

    elif dataset.lower() == 'pbc':
        features, t, e = datasets.load_dataset('PBC')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})

    elif dataset.lower() == 'flchain':
        features, outcomes = load_flchain()
        outcomes = pd.DataFrame(outcomes).astype(np.float32)
        outcomes = outcomes.rename(columns={'futime':'time','death':'event'})
        outcomes['time'] = outcomes['time'] + 1e-10
        features = features.drop(columns=['sample.yr'])
        cat_feats = ['chapter', 'flc.grp', 'mgus', 'sex']
        num_feats = [key for key in features.keys() if key not in cat_feats]
        features = preprocessing.Preprocessor().fit_transform(
            cat_feats=cat_feats,
            num_feats=num_feats,
            data=features,
            )
    elif dataset.lower() == 'support_pycox':
            f1 = h5py.File(
                './auton_lab/auton_survival/datasets/support_train_test.h5',
                'r+'
                )
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
            
            cat_feats = [
                'sex', 
                'race', 
                'presence of diabetes', 
                'presence of dementia', 
                'presence of cancer'
                ]
            
            num_feats = [key for key in features.keys() if key not in cat_feats]
            
            features = preprocessing.Preprocessor().fit_transform(
            cat_feats=cat_feats,
            num_feats=num_feats,
            data=features,
            )
        
    elif dataset.lower() == 'metabric_pycox':
        f1 = h5py.File(
            './auton_lab/auton_survival/datasets/metabric_IHC4_clinical_train_test.h5',
            'r+'
            )
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
        
        feature_names = ['x{}'.format(i) for i in range(x.shape[1])]
        
        outcomes_['event'] = e
        outcomes_['time'] = t
        for i, key in enumerate(feature_names):
            features_[key] = x[:,i]
        outcomes = pd.DataFrame.from_dict(outcomes_)
        features = pd.DataFrame.from_dict(features_)
        
        cat_feats = [
            'x4', 
            'x5', 
            'x6', 
            'x7', 
            ]
        
        num_feats = [key for key in features.keys() if key not in cat_feats]
        
        features = preprocessing.Preprocessor().fit_transform(
        cat_feats=cat_feats,
        num_feats=num_feats,
        data=features,
        )

    outcomes.time = outcomes.time + 1e-15

    if scale_time:
        outcomes.time = (
            outcomes.time - outcomes.time.min()
            ) / (
                outcomes.time.max() - outcomes.time.min()
                ) + 1e-15

    return outcomes, features

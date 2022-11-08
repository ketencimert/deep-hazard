# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:47:07 2022

@author: Mert
"""
import numpy as np
import pandas as pd

from pycox.datasets import flchain
from pycox.datasets import support
from pycox.datasets import metabric
from pycox.datasets import gbsg

import torch

from auton_lab.auton_survival import datasets, preprocessing

from simulation_studies.synthetic_data import SyntheticData  

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

        features = support.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])
        features_ = dict()

        feature_names = [
            'age',
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
            "serum’s creatinine",
             ]

        for i, key in enumerate(feature_names):
            features_[key] = features.iloc[:,i]
        features = pd.DataFrame.from_dict(features_)

        cat_feats = [
            'sex',
            'race',
            'presence of diabetes',
            'presence of dementia',
            'presence of cancer'
            ]
        num_feats = [
            key for key in features.keys() if key not in cat_feats
            ]

    elif dataset.lower() == 'flchain':
        features = flchain.read_df()
        outcomes = features[['death', 'futime']]
        outcomes = outcomes.rename(
            columns={'death':'event', 'futime':'time'}
            )
        features = features.drop(columns=['death', 'futime'])

        cat_feats = ['flc.grp', 'mgus', 'sex']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'metabric':
        features = metabric.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x4', 'x5', 'x6', 'x7']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'gbsg':
        features = gbsg.read_df()
        outcomes = features[['event', 'duration']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'duration':'time'}
            )
        features = features.drop(columns=['event', 'duration'])

        cat_feats = ['x0', 'x1', 'x2']
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'pbc':
        features, t, e = datasets.load_dataset('PBC')
        features = pd.DataFrame(features)
        outcomes = pd.DataFrame([t,e]).T
        outcomes = outcomes.rename(columns={0:'time',1:'event'})

        features = pd.DataFrame(features)
        cat_feats = []
        num_feats = [key for key in features.keys() if key not in cat_feats]

    elif dataset.lower() == 'aki/ckd':

        data = pd.read_csv(
            './columbia-aki-ckd/aki-ckd-filtered_use_{}.csv'.format(
                dataset.lower().split('_')[-1]
                )
            )
        drop = [x for x in data.keys() if 'unnamed' in x.lower()]
        data = data.drop(columns = drop)

        # data = data[~data['time'].isna()]

        #let's not use these "pseudo-categorical variables in our experiments"
        pseudo_categorical = [
            key for key in data.keys() if 'categorical' in key
            ]

        data = data.drop(
            columns=[
                'person_id',
                'beg_date',
                'event_date',
                'cohort_end_date'
                ] + pseudo_categorical
            )

        data = data.sample(frac=1).reset_index(drop=True)

        outcomes = data[['time', 'event']]
        features = data[[x for x in data.keys() if x not in outcomes.keys()]]

        features = features[~outcomes.time.isna()]
        outcomes = outcomes[~outcomes.time.isna()]

        num_feats = [key for key in features.keys() if 'numerical' in key]
        num_feats.append('age')
        cat_feats = list(set(features.keys()) - set(num_feats))
    elif dataset.lower() == 'synthetic':
        features, num_feats = SyntheticData()
        outcomes = features[['event', 'time']]
        outcomes = outcomes.rename(
            columns={'event':'event', 'time':'time'}
            )
        features = features.drop(columns=['event', 'time'])
        cat_feats = []
    if dataset.lower() != 'synthetic':
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

class SurvivalData(torch.utils.data.Dataset):
    def __init__(self, x, t, e, bs, device, dtype=torch.double):
        self.bs = bs
        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(t, dtype=dtype),
                torch.tensor(e, dtype=dtype)
            ] for x, t, e in zip(x, t, e)
        ]

        self.device = device
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.device:
                self._cache[index][0] = self._cache[
                    index][0].to(self.device)

                self._cache[index][1] = self._cache[
                    index][1].to(self.device)

                self._cache[index][2] = self._cache[
                    index][2].to(self.device)

        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_

    def __blen__(self):
        return int(np.ceil(self.__len__() / self.bs))